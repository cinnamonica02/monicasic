# sim/ - C++ ISS and memory model

The first slice is a dependency-free C++17 instruction-set reference for the
toy-MoE route/dispatch/combine path. It uses the integer rounding rule planned
for hardware and is tested by CTest in CI.

Python (`tiny_moe/`) stays the home of NumPy/model references. The static
schedule exporter and ISS live here once they no longer need Python/NumPy.

```text
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```
