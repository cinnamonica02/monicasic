# 05 — MatX One style (aspirational reference, from INSPO.md)

Source: `INSPO.md` chat transcript. These claims come from an AI chat plus MatX's public statements. They are **unverified** and MatX has published no microarchitecture paper (per the chat itself).
Use it for the idea only: splittable array, weights in SRAM, KV cache in bulk memory.

```mermaid
flowchart TB
    subgraph CHIP["MatX One (as described)"]
        subgraph ARRAY["Splittable systolic array"]
            ONE["one big array<br/>dense layers"]
            SPLIT["or partitioned sub-arrays"]
            S1["expert A"]
            S2["expert B"]
            S3["attention"]
            S4["decode stream"]
            SPLIT --> S1
            SPLIT --> S2
            SPLIT --> S3
            SPLIT --> S4
        end
        SRAMW["SRAM: ALL model weights resident<br/>(low latency, high tokens/s)"]
        STREAM["Streaming FIFOs"]
    end

    HBM["HBM: KV cache<br/>(long context)"]
    NET["Scale-up / scale-out interconnect<br/>experts spread across chips"]

    SRAMW --> ARRAY
    ARRAY <--> STREAM
    STREAM <--> HBM
    ARRAY <--> NET
```

What we take from it at our scale: the weights-in-SRAM and KV-elsewhere split, and the array that can be partitioned per expert.
We drop HBM and the multi-chip interconnect, since we are one small die.
