# llm.🔥

This project is a port of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) to [Mojo](https://docs.modular.com/mojo), started by @dorjeduck. It is under active development and subject to significant changes. Users should expect to encounter bugs and unfinished features. Not to be taken too serious yet, an early version of someone just learning Mojo ...

## Implementation

- **train_gpt_basic.mojo**: Basic port of `train_gpt.c` to Mojo. Does not leverage Mojo's capabilities.
- **train_gpt.mojo**: Enhanced version utilizing Mojo's performance gems like vectorization and parallelization. Work in progress.

## How to Use

1. Set up `llm.c` by following the instructions at [llm.c](https://github.com/karpathy/llm.c).
2. Place `train_gpt.mojo` and `train_gpt_basic.mojo` (if you have time) in the `llm.c` directory.
3. Run `mojo train_gpt.mojo`.

```bash
# If it complains about this then set it to your python3 library path.
export MOJO_PYTHON_LIBRARY=/opt/homebrew/opt/python@3.12/Frameworks/Python.framework/Versions/3.12/lib/libpython3.12.dylib
```


## Benchmarks

Preliminary benchmark results:

| Implementation             | Average Training Loop Time |
|----------------------------|----------------------------|
| train_gpt2.c (with OpenMP)  | 2119 ms                    |
| train_gpt2.mojo             | 2346 ms                    |
| train_gpt2.c  (no OpenMP)   | 7473 ms                    |
| train_gpt2_basic.mojo       | 54509 ms                   |

!['Training Loop Times'](imgs/training_loop_times_chart.png)

## Development Roadmap

- **Implementation Improvement**: Enhance `train_gpt.mojo` to fully exploit Mojo's capabilities, including further optimization for speed and efficiency.
- **Port test_gpt2.c to Mojo**: Coming soon
- **Following Changes of llm.c**: Regularly update the Mojo port to align with the latest improvements and changes made to `llm.c`.
- **Solid Benchmarks**: Develop comprehensive and reliable benchmarks to accurately measure performance improvements and compare them against other implementations.

## License

MIT
