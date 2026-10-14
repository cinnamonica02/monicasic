# hw/ — RTL and hardware verification (empty for now)

Planned layout, following the reference repos:

```
hw/
  rtl/    SystemVerilog sources          (tiny-tpu: src/, NVDLA: hw/vmod)
  tb/     cocotb / Verilator testbenches (tiny-tpu: test/, NVDLA: hw/verif)
  syn/    OpenLane / LibreLane configs and reports  (NVDLA: hw/syn)
  fpga/   board constraints and build scripts
```

Nothing here until the integer reference (`tiny_moe/ref/`) and `docs/spec/NUMERICS_CONTRACT.md` are frozen. The RTL is checked bit-exact against that integer reference (A3 in `docs/spec/ACCEPTANCE.md`).
