"""
Collapses nsys GPU stacks to be used in a FlameGraph.

nsys profile --capture-range cudaProfilerApi ./train_gpt2cu


"""

from collections.abc import Iterator
from typing import TextIO

import argparse
import copy
import dataclasses
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile

import pandas as pd
import tqdm

_PARSE_NSYS_EVENTS_SQL = """
WITH processed_cuda_kernels as (
SELECT
	null as nvtx_name,
	CUPTI_ACTIVITY_KIND_RUNTIME.`start` as `cpu_start`,
	CUPTI_ACTIVITY_KIND_RUNTIME.`end` as `cpu_end`,
	"kernel_launch" as kind,
	COALESCE(CUPTI_ACTIVITY_KIND_KERNEL.`start`,
	CUPTI_ACTIVITY_KIND_MEMCPY.`start`) as `kernel_start`,
	COALESCE(CUPTI_ACTIVITY_KIND_KERNEL.`end`,
	CUPTI_ACTIVITY_KIND_MEMCPY.`end`) as `kernel_end`,
	IIF(CUPTI_ACTIVITY_KIND_KERNEL.`start` is not null, demangled_name.value, 'memcpy') as demangled_name
FROM
	CUPTI_ACTIVITY_KIND_RUNTIME
LEFT JOIN
	CUPTI_ACTIVITY_KIND_KERNEL
ON
	CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_KERNEL.correlationId
LEFT JOIN StringIds as demangled_name
ON
	CUPTI_ACTIVITY_KIND_KERNEL.demangledName = demangled_name.id
LEFT JOIN CUPTI_ACTIVITY_KIND_MEMCPY
ON
	CUPTI_ACTIVITY_KIND_RUNTIME.correlationId = CUPTI_ACTIVITY_KIND_MEMCPY.correlationId
),
processed_nvtx_events AS (
SELECT
	NVTX_EVENTS.text as nvtx_name,
	NVTX_EVENTS.`start` as `cpu_start`,
	NVTX_EVENTS.`end` as `cpu_end`,
	ENUM_NSYS_EVENT_TYPE.label as kind,
	null as `kernel_start`,
	null as `kernel_end`,
	null as demangled_name
FROM
	NVTX_EVENTS
LEFT JOIN ENUM_NSYS_EVENT_TYPE
ON
	NVTX_EVENTS.eventType = ENUM_NSYS_EVENT_TYPE.id
WHERE
	NVTX_EVENTS.eventType = 59
	-- NvtxPushPopRange
),
joined_nvtx_and_cuda_events AS (
SELECT
	*
FROM
	processed_cuda_kernels
UNION ALL
SELECT
	*
FROM
	processed_nvtx_events
)
SELECT
	*
FROM
	joined_nvtx_and_cuda_events
ORDER BY
	`cpu_start`;
"""

# Determines how we should sort if a timestamp is identical (ascending)
_EVENT_KIND_EVENT_TYPE_TO_SORT_KEY = {
    ("NvtxPushPopRange", "start"): 0,
    ("kernel_launch", "start"): 1,
    ("kernel_launch", "end"): 2,
    ("NvtxPushPopRange", "end"): 3,
}


@dataclasses.dataclass(frozen=True)
class FlamegraphEvent:
    stack_trace: list[str]
    gpu_time_ns: float


def _export_nsys_rep_to_sqlite(
    nsys_executable: str, report_path: pathlib.Path, output_path: pathlib.Path
):
    nsys_result = subprocess.run(
        [
            nsys_executable,
            "export",
            "--type=sqlite",
            "--output",
            os.fspath(output_path),
            os.fspath(report_path),
        ]
    )
    nsys_result.check_returncode()


def _nsys_rep_as_pandas_df(sqlite_output_path: pathlib.Path) -> pd.DataFrame:
    connection = sqlite3.connect(os.fspath(sqlite_output_path))
    return pd.read_sql_query(_PARSE_NSYS_EVENTS_SQL, connection)


def _joined_row_to_secondary_sort_key(row):
    event_kind = row.kind
    event_type = row.cpu_event_type
    return _EVENT_KIND_EVENT_TYPE_TO_SORT_KEY[(event_kind, event_type)]


def _nsys_pandas_df_to_flat_cpu_events(df):
    # Add GPU time for all events
    df["gpu_time_ns"] = df["kernel_end"] - df["kernel_start"]

    # Split each CPU events into two: start and end
    # We will then sort by the event time to reconstruct a stack trace.
    df_cpu_start_events = df.copy()
    df_cpu_start_events["cpu_event_time"] = df_cpu_start_events["cpu_start"]
    df_cpu_start_events["cpu_event_type"] = "start"

    df_cpu_end_events = df.copy()
    df_cpu_end_events["cpu_event_time"] = df_cpu_end_events["cpu_end"]
    df_cpu_end_events["cpu_event_type"] = "end"

    # Sort by the event time. In case of equal event times, sort by the secondary key.
    joined_events = pd.concat((df_cpu_start_events, df_cpu_end_events))
    joined_events["secondary_sort_key"] = joined_events.apply(
        _joined_row_to_secondary_sort_key, axis=1
    )
    joined_events = joined_events.sort_values(
        by=["cpu_event_time", "secondary_sort_key"], ascending=True
    )
    return joined_events


def _fold_stacks(events_df: pd.DataFrame) -> Iterator[FlamegraphEvent]:
    joined_events = _nsys_pandas_df_to_flat_cpu_events(events_df)

    current_stack = []
    for _, row in tqdm.tqdm(
        joined_events.iterrows(),
        total=joined_events.shape[0],
        desc="Processing stack events",
    ):
        cpu_event_type = row.cpu_event_type

        current_stack_trace_name = None
        if row.kind == "kernel_launch":
            assert row.demangled_name is not None, f"Bad row: {row}"
            current_stack_trace_name = row.demangled_name
        elif row.kind == "NvtxPushPopRange":
            assert row.nvtx_name is not None, f"Bad row: {row}"
            current_stack_trace_name = row.nvtx_name

        if cpu_event_type == "start":
            current_stack.append(current_stack_trace_name)

            if row.kind == "kernel_launch":
                # Push a new FlameGraph event, reached final point
                gpu_time_ns = row.gpu_time_ns
                if pd.isnull(gpu_time_ns):
                    # Weird event, skip it
                    continue

                yield FlamegraphEvent(
                    stack_trace=copy.copy(current_stack),
                    gpu_time_ns=gpu_time_ns,
                )

        elif cpu_event_type == "end":
            assert (
                current_stack and current_stack[-1] == current_stack_trace_name
            ), f"Bad stack: {current_stack}, got end name {current_stack_trace_name}"
            current_stack.pop()
        else:
            assert False, f"Unknown event type: {cpu_event_type} {row}"

    assert (
        not current_stack
    ), f"Got bad stack at the end of all processing: {current_stack}"


def _filter_flamegraph_events(
    events: Iterator[FlamegraphEvent],
    strip_layer_number: bool = True, # Do not show "Layer N" in the stack trace
    strip_train_iter_number: bool = True, # Do not show "Train step N" in the stack trace
    filter_out_validation: bool = True, # If the call originates from the validation, remove the event
) -> Iterator[FlamegraphEvent]:
    for event in events:
        stack_trace = event.stack_trace
        if filter_out_validation and any('validation' in x for x in event.stack_trace):
            continue

        if strip_layer_number:
            stack_trace = [x for x in stack_trace if not x.startswith('Layer')]

        if strip_train_iter_number:
            stack_trace = [x for x in stack_trace if not x.startswith('Train step')]

        yield dataclasses.replace(event, stack_trace=stack_trace)


def _print_flamegraph_events(
    events: Iterator[FlamegraphEvent],
    out_file: TextIO,
):
    for event in events:
        stack_trace_str = ';'.join(x.replace(';', '_') for x in event.stack_trace)
        gpu_time_ns = int(event.gpu_time_ns)
        print('{} {}'.format(stack_trace_str, gpu_time_ns), file=out_file)


def _fold_and_print_filtered_stacks(
    nsys_executable: str,
    report_path: pathlib.Path,
    strip_layer_number: bool = True, # Do not show "Layer N" in the stack trace
    strip_train_iter_number: bool = True, # Do not show "Train step N" in the stack trace
    filter_out_validation: bool = True, # If the call originates from the validation, remove the event
):
    with tempfile.TemporaryDirectory() as temp_dir:
        sqlite_path = pathlib.Path(temp_dir) / "db.sqlite"
        _export_nsys_rep_to_sqlite(nsys_executable, report_path, sqlite_path)
        events_df = _nsys_rep_as_pandas_df(sqlite_path)
        joined_events = _nsys_pandas_df_to_flat_cpu_events(events_df)
        flamegraph_events_iter = _fold_stacks(joined_events)
        filtered_flamegraph_events_iter = _filter_flamegraph_events(flamegraph_events_iter)
        _print_flamegraph_events(filtered_flamegraph_events_iter, sys.stdout)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Collapse NSight Systems GPU events for a FlameGraph"
    )
    parser.add_argument(
        "report_path",
        type=pathlib.Path,
        help="Path to the nsys profile report file (reportN.nsys-rep).",
    )
    parser.add_argument(
        "--nsys_executable", default="nsys", help="nsys executable to run nsys export"
    )

    parser.add_argument(
        "--no_strip_layer_number",
        action="store_true",
        default=False,
        help="Show 'Layer N' in the stack trace"
    )
    parser.add_argument(
        "--no_strip_train_iter_number",
        action="store_true",
        default=False,
        help="Show 'Train step N' in the stack trace"
    )
    parser.add_argument(
        "--no_filter_out_validation",
        action="store_true",
        default=False,
        help="Keep the stack, even if the call originates from the validation"
    )

    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        _fold_and_print_filtered_stacks(
            nsys_executable=args.nsys_executable,
            report_path=args.report_path,
            strip_layer_number=not args.no_strip_layer_number,
            strip_train_iter_number=not args.no_strip_train_iter_number,
            filter_out_validation=not args.no_filter_out_validation,
        )
    except AssertionError as e:
        raise ValueError('Got an assertion. A missing NVTX context name, or a broken stack can happen if NSight Systems subsamples events. This happens if you run nsys profile for too long. Try running only for the first 10 steps.') from e


if __name__ == "__main__":
    main()
