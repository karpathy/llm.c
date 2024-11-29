# Profiling with NSight System.

## Installation

See this manual: https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html.

On a server with a GPU, you need to download the `CLI Only` package. We don't need to install the UI.

To view the profiler results on your desktop/laptop, download and install the `<Windows/Linux/MacOS> host` package.

## Running NSight Systems

Here is how you can run the entire training under NSight Systems profiler.

```
nsys profile ./train_gpt2cu
```

Alternatively, you can profile only for the first 10 steps (useful for FlameGraph generation):
```
nsys profile --capture-range cudaProfilerApi ./train_gpt2cu
```

`cudaProfilerApi` as a capture range tells `nsys` to profile only after a `cudaProfilerStart` call, and finish profiling after `cudaProfilerStop`.

This will generate a report named `reportN.nsys-rep`.

## Viewing the report in NSight Systems GUI

Copy the report file to your local machine. You can use `scp` to do that:
```
scp <SSH username>@<SSH ip address>:<absolute path to the report file on a server> <local path>
```

After that, you can view the report in the UI. Click "File -> Open" in the upper menu and select the downloaded report file.

### Understanding the NSight Systems GUI

There are two primary timelines:
1. CPU timeline, where our CPU launches kernels for the GPU.
2. GPU timeline, where our GPU executes the kernels.

When analyzing the CPU timeline, you should be looking for periods of time when the CPU doesn't schedule the kernels for the GPU.
For example: a `cudaStreamSynchronize` call blocks CPU execution. This leads to a GPU stall right after the CPU is unblocked, because CPU needs some time to add new kernels. Look for ways to remove such stalls.

When analyzing the GPU timeline, look for kernels that take the most time. Try to think if you can reduce the execution time of some of them.

## Converting nsys profile to a FlameGraph

WARNING: This is experimental and not very well tested, may break or report incorrect results. Cross-check with the NSight Systems GUI.

First, profile only the first 10 steps of training:
```
nsys profile --capture-range cudaProfilerApi ./train_gpt2cu
```

If you profile longer, `nsys` starts dropping NVTX ranges due to a large report size. This leads to broken stacks.

Then, download `flamegraph.pl` file from the official FlameGraph GitHub repository:
```
wget https://raw.githubusercontent.com/brendangregg/FlameGraph/master/flamegraph.pl
```

Run the following command:
```
python ./dev/tools/collapse_nsys_gpu_stacks.py <path to .nsys-rep report file> | perl flamegraph.pl > flame_graph.svg
```

If you encounter errors when importing `pandas`, run this:
```
python -m pip install pandas
```

Download the resulting `flame_graph.svg` file and open it in the browser. The stacks are clickable.
If you highlight, FlameGraph will report how many "Samples" hit this stack. In our case, it's the total number of nanoseconds spend in this stack.

Here are the command line arguments for `collapse_nsys_gpu_stacks.py`:
```
usage: collapse_nsys_gpu_stacks.py [-h] [--nsys_executable NSYS_EXECUTABLE] [--no_strip_layer_number] [--no_strip_train_iter_number] [--no_filter_out_validation] report_path

Collapse NSight Systems GPU events for a FlameGraph

positional arguments:
  report_path           Path to the nsys profile report file (reportN.nsys-rep).

options:
  -h, --help            show this help message and exit
  --nsys_executable NSYS_EXECUTABLE
                        nsys executable to run nsys export
  --no_strip_layer_number
                        Show 'Layer N' in the stack trace
  --no_strip_train_iter_number
                        Show 'Train step N' in the stack trace
  --no_filter_out_validation
                        Keep the stack, even if the call originates from the validation
```