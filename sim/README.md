# sim/ — C++ simulator and memory model (empty for now)

Planned: an instruction/command-stream simulator that matches the integer reference bit-for-bit, plus a DRAM-bandwidth model used to predict A4 before any FPGA build. Add `sim/CMakeLists.txt` and the root `CMakeLists.txt` will pick it up.

Python (`tiny_moe/`) stays the home of the references and the exporter. Rust, if used, would get its own top-level `rust/` directory with a `Cargo.toml` (cargo is installed locally; nothing is created yet).
