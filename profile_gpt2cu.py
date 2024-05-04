# runs profiling with ncu, generates a `profile.ncu-rep` for viewing with NSight Compute, and prints out
# basic kernel stats.
# Note: If you run into errors because of missing access rights to performance counters, try
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag

import subprocess
import csv
from collections import defaultdict
import shutil

# find ncu: Is it on PATH?
NCU = shutil.which("ncu")
# otherwise, guess a standard location
if NCU is None:
    NCU = "/usr/local/cuda/bin/ncu"

# build the exe
subprocess.check_call(["make", "profile_gpt2cu"])

# record metrics
# --full and --import-source are entirely superfluous for this script, but you might want to
# manually inspect `profile.ncu-rep`, so we keep it here
cmd = [NCU, "--set", "full", "--import-source", "yes", "-o", "profile", "-f", "./profile_gpt2cu"]
subprocess.check_call(cmd)

# generate csv
# https://forums.developer.nvidia.com/t/converting-nsys-rep-file-into-a-csv-file-with-formatting-like-the-summary-page-in-ncu-gui/231717/3
metrics = [
    "gpu__time_duration.sum",                   # total time
    "dram__bytes_read.sum",                     # DRAM reads
    "dram__bytes_write.sum",                    # DRAM writes
    "lts__t_sectors_srcunit_tex_op_read.sum",   # L2 reads (sectors -- 32B)
    "lts__t_sectors_srcunit_tex_op_write.sum",  # L2 reads (sectors -- 32B)
    "smsp__inst_executed.sum",                   # instructions
]
cmd = [NCU, "-i", "profile.ncu-rep", "--csv", "--page", "raw", "--metrics", ",".join(metrics)]
result = subprocess.check_output(cmd, text=True).strip()

reader = csv.reader(result.splitlines(keepends=True))

# model config
CLS_START = 15
CLS_NUM = 6
ADAM_ID = 44
N_LAYERS = 12

summaries = defaultdict(lambda: 0.0)
passes = defaultdict(lambda: 0.0)
total = defaultdict(lambda: 0.0)
no_cutlass = 0.0
CC = ""

print()
print("Kernel calls:")
for rid, row in enumerate(reader):
    if rid == 0:
        #  headings
        print(f"id pass {'name':<40} {'time':>8} {'RAM rd':>8} {'RAM wt':>8} {'L2 rd':>8} {'L2 wt':>8} {'inst':>8}")
        continue
    if rid == 1:
        # units
        units = f"        {'':<40} {'ms':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'MInst':>8}"
        print(units)
        print("." * len(units))
        continue
    if rid == 2:

        CC = row[10]

    # actual data
    kernel = row[4]
    time = float(row[13])
    read = float(row[11])
    write = float(row[12])
    l2_read = float(row[14])
    l2_write = float(row[15])
    inst = float(row[16]) / 1e6

    kid = rid - 2

    if kid == 0 or kid == ADAM_ID - 1:
        pass_name = "enc"
    elif CLS_START <= kid < CLS_START + CLS_NUM:
        # the classifier part, counts only once
        pass_name = "cls"
    elif kid == ADAM_ID:
        # encoder layer or adam
        pass_name = "opt"
    else:
        pass_name = "fwd" if kid < CLS_START else "bwd"
        time *= N_LAYERS
        read *= N_LAYERS
        write *= N_LAYERS
        l2_read *= N_LAYERS
        l2_write *= N_LAYERS

    # split at "(" -- argument list
    fn_name = kernel.split("(")[0]
    # some names include the return value, others don't?
    if " " in fn_name:
        fn_name = fn_name.split(" ")[1]
    if "cutlass" in fn_name:
        fn_name = fn_name.split("<")[0]
        pass
    else:
        no_cutlass += time

    # convert L2 to GiB
    l2_read = l2_read * 32 / 1024 / 1024 / 1024
    l2_write = l2_write * 32 / 1024 / 1024 / 1024

    summaries[fn_name] += time
    passes[pass_name] += time
    total['time'] += time
    total['read'] += read
    total['write'] += write
    total['l2_read'] += l2_read
    total['l2_write'] += l2_write
    total['inst'] += inst

    print(f"{kid:02} {pass_name:4} {fn_name:<40} {time:8.2f} {read:8.2f} {write:8.2f} {l2_read:8.2f} {l2_write:8.2f} {inst:8.2f}")

total_time = total['time']
print("." * len(units))
print(f"        {'Total':<40} {total['time']:8.2f} {total['read']:8.2f} {total['write']:8.2f} {total['l2_read']:8.2f} {total['l2_write']:8.2f} {total['inst']:8.2f}")

print()
print("Kernel type summaries:")
print(f"  {'name':<40} {'time':>6} {'frac':>6}")
ordered = sorted(summaries.items(), key=lambda x: x[1], reverse=True)
for entry, value in ordered:
    print(f"  {entry:<40} {value:6.2f} {100*value / total_time:6.2f}%")


ts = total_time / 1000
summary = f"""
In total, a training step takes {total_time:.1f}ms, distributed as:
  {passes['enc']:.1f}ms ({100 * passes['enc'] / total_time:.1f}%) in the encoder,
  {passes['fwd']:.1f}ms ({100 * passes['fwd'] / total_time:.1f}%) in forward blocks,
  {passes['cls']:.1f}ms ({100 * passes['cls'] / total_time:.1f}%) in the classifier part,
  {passes['bwd']:.1f}ms ({100 * passes['bwd'] / total_time:.1f}%) in backward blocks, and
  {passes['opt']:.1f}ms ({100 * passes['opt'] / total_time:.1f}%) in the optimizer.

We read {total['read']:.1f}GiB ({total['read']/ts:.1f}GB/s) and write {total['write']:.1f}GiB ({total['write']/ts:.1f}GB/s) to DRAM,
read {total['l2_read']:.1f}GiB ({total['l2_read']/ts:.1f}GB/s) and write {total['l2_write']:.1f}GiB ({total['l2_write']/ts:.1f}GB/s) to L2,
and execute {total['inst'] / 1000:.1f} billion instructions ({total['inst'] / 1000 / ts:.1f} GInst/s).
"""
print(summary)