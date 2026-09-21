# perf-tools-orchestrator (recovery backup)

`benchmark_smart_kiosk_v2v.py` in this directory is **not** the copy that
runs. The live copy lives inside the `performance-tools` submodule at
`performance-tools/benchmark-scripts/benchmark_smart_kiosk_v2v.py`, so it can
bring the Smart Kiosk stack up the same way `benchmark_order_accuracy.py`
does for the vision pipelines, and chain straight into that directory's
`consolidate_multiple_run_of_metrics.py` / `usage_graph_plot.py` afterwards.

That live copy is **untracked inside the submodule** (it has not been
upstreamed into `intel-retail/performance-tools`), so `make
update-submodules` resets the submodule's working tree to the pinned
upstream commit and wipes it whenever the pinned commit changes. The copy in
this directory exists purely as a recovery source: `make update-submodules`
automatically reinstalls it into the submodule after every update (see the
Makefile), and it can also be restored manually with:

```
cp tests/benchmarks/perf-tools-orchestrator/benchmark_smart_kiosk_v2v.py \
   ../performance-tools/benchmark-scripts/benchmark_smart_kiosk_v2v.py
```

**If you change the orchestrator's behavior, edit the submodule's copy and
then copy the result back here** so this backup doesn't go stale. The two
files must stay byte-identical.
