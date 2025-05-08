## Architecture Guide

### Introduction
traiNNer-redux supports several super resolution and restoration architectures/networks. This page introduces the most popular and recommended architectures to train. The architecture should be chosen depending on your requirements for speed, VRAM consumption, and quality.

Note that an architecture not being listed on this page does not mean it's not recommended. Not all architectures have been thoroughly tested. For a complete list of supported architectures, see the [Architecture reference](/arch_reference.html). For benchmarks, see the [benchmark charts](/benchmark_charts.html).

### Recommendations
- Ultra light-weight
   - `SuperUltraComapct`
- Light-weight
   - `SPAN_S`
- Medium-weight
   - `RealPLKSR`
- Medium heavy-weight
   - `SwinIR M`
- Heavy-weight
   - `DAT2`
- Ultra heavy-weight
   - `HAT_L`, `DRCT_L`, `ATD`
