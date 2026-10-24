# tests/

Run with `pytest tests/` (add `-m "not slow"` to skip the end-to-end runs).

| Kind | Files |
|---|---|
| Unit tests | `test_core.py`, `test_models.py`, `test_constraints.py`, `test_nn_layers.py`, `test_datapipes.py`, `test_cache.py`, `test_splits.py`, `test_metrics.py`, `test_training_utils.py`, `test_registry.py`, `test_profiling.py`, `test_tuner.py`, `test_integration.py` |
| Distributed (2 CPU processes, gloo) | `test_distributed_multirank.py`, `test_mappings_multirank.py`, `test_ddp_example.py`, `test_checkpoint_distributed.py`, plus single-process `test_distributed*.py` |
| Optional-dependency / GPU tests | `test_export.py` (needs the `export` extra); fp16 and graph-capture tests skip without a GPU |
| End-to-end (`slow`) | `test_projects_smoke.py` — every project pipeline and example with tiny settings |
