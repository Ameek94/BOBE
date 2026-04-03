# BOBE Code Cleanup: Redundancy Removal & Results Simplification

## Context

The BOBE codebase has grown organically. Several incremental refactors have left behind dead code, near-duplicate convergence methods, and a fragile checkpoint split (flow weights saved to a separate `.pkl` from the main state). The `BOBEResults` class tracks data in many parallel lists that are tedious to maintain and serialise. This plan addresses these issues while preserving the public API.

---

## 1. Dead Code in `bo.py`

### 1a. Double initialisation of result containers (`run()`)
**Lines 1346–1347 and 1426–1427** both do:
```python
self.samples_dict = {}
self.results_dict = {}
```
**Fix**: Remove the first duplicate pair at lines 1346–1347.

### 1b. Unused `n_points_since_last_ns` counter
`self.n_points_since_last_ns = 0` is set in `run()` (line 1360) but never incremented or read anywhere in the loop.
`self.ns_n_points` is logged and stored but never triggers any mid-run nested sampling (the only NS call is the `do_final_ns` block at the very end).
**Fix**: Remove both `self.n_points_since_last_ns` and `self.ns_n_points`, and remove `ns_n_points` from `get_dimension_based_defaults()`, `run()` signature, and the settings dict update.

### 1c. Unenforced `min_evals` in `run_EI`
`run_EI` has a commented-out guard `# if current_evals >= self.min_evals:` (line 1491). The parameter is defined, logged, and dimension-scaled but has no effect.
**Fix**: Either enforce the guard (uncomment it) or remove `min_evals` from `run_EI`'s logic entirely. Decide which is intended.

### 1d. Redundant `load_gp_file` / `load_gp_statedict` helpers
Both functions in `bo.py` (lines 26–66) do the same conditional dispatch:
```python
if clf:  return GPwithClassifier.load/from_state_dict(...)
else:    return GP.load/from_state_dict(...)
```
**Fix**: Merge into one function `_load_gp(source, clf)` that accepts either a filename or a state dict.

---

## 2. Convergence Method Consolidation (`bo.py`)

There are three convergence checkers with duplicated counter/logging logic:

| Method | Used by | Logic |
|---|---|---|
| `check_convergence_ei` | `run_EI` | acq value vs threshold |
| `_check_convergence` | `run_weighted_integrated_posterior` | acq value + KL divergence |
| `check_convergence_logz` | `do_final_ns` block only | logz delta + KL + min_delta checkpoint |

All three share: incrementing `self.convergence_counter`, comparing against `self.convergence_n_iters`, logging progress/success, and returning `bool`.

**Fix**: Extract a shared `_advance_convergence_counter(condition: bool) -> bool` helper:
```python
def _advance_convergence_counter(self, condition: bool) -> bool:
    if condition:
        self.convergence_counter += 1
        if self.convergence_counter >= self.convergence_n_iters:
            return True
        log.info(f"Convergence iter {self.convergence_counter}/{self.convergence_n_iters}")
    else:
        self.convergence_counter = 0
    return False
```
Each checker calls this instead of re-implementing the pattern. The criterion-specific logic (EI log-transform, KL computation, logz delta) stays in each method.

---

## 3. `results_dict` Partial Population During Loop

Throughout `run_EI` and `run_weighted_integrated_posterior`, there are scattered:
```python
self.results_dict['termination_reason'] = self.termination_reason
self.results_dict['logz'] = logz_dict
```
…but `finalise_results()` completely rebuilds `self.results_dict` from scratch at the end anyway. The mid-loop partial updates are confusing and fragile (if `finalise_results()` is called, the partial updates are silently overwritten).

**Fix**: Remove all `self.results_dict['...'] = ...` assignments from within the loops. They set `self.termination_reason` only on `self` (which `finalise_results()` already reads). Keep only the `self.termination_reason = ...` assignments.

---

## 4. Checkpoint Split: Flow Weights in Separate File

In `_save_checkpoint()` (line 428–429):
```python
if isinstance(self.transform, NormalisingFlowTransform) and self.transform._use_flow:
    self.transform.save(self.save_path)  # → saves a separate _flow.pkl
```
And in `_handle_resume()` (lines 450–458), the flow is loaded from a second file. This means a complete run checkpoint is split across two files, which is fragile (one can be deleted without the other).

**Fix**: Serialise the flow state dict directly into the main checkpoint pkl. `NormalisingFlow.state_dict()` already uses equinox and returns a picklable structure. In `_save_checkpoint`, add `'flow_state': self.transform._flow.state_dict() if ... else None` to the checkpoint dict. In `_handle_resume`, reconstruct from `data.get('flow_state')`. Remove the separate `.save()`/`.load()` file path.

---

## 5. BOBEResults: Parallel Tracking Lists → List of Records

`BOBEResults._initialize_fresh()` creates **13 separate lists** that are conceptually grouped:

```
acquisition_iterations, acquisition_values, acquisition_functions
gp_iterations, gp_lengthscales, gp_kernel_variances
best_loglike_iterations, best_loglike_values
kl_iterations, kl_divergences, successive_kl
convergence_history (already a list of ConvergenceInfo)
logz_evolution
```

Each group is serialised, deserialised, and restored in parallel in `get_state_dict()` / `restore_from_checkpoint()`. Any rename or addition requires changes in 3+ places.

**Fix**: Consolidate each logical group into a list of records (simple dicts or lightweight dataclasses, consistent with the existing `ConvergenceInfo` pattern):

```python
# Instead of three parallel lists:
self.acquisition_history: List[dict] = []
# Each entry: {'iteration': int, 'value': float, 'function': str}

self.gp_history: List[dict] = []
# Each entry: {'iteration': int, 'lengthscales': list, 'kernel_variance': float}

self.best_loglike_history: List[dict] = []
# Each entry: {'iteration': int, 'best_loglike': float}

self.kl_history: List[dict] = []
# Each entry: {'iteration': int, 'successive': dict}
```

Update `update_acquisition()`, `update_gp_hyperparams()`, `update_best_loglike()`, `update_kl_divergences()` to append dicts. Update `get_state_dict()` and `restore_from_checkpoint()` to serialise/restore the new structure. Getter methods (`get_acquisition_data()`, `get_gp_data()`, etc.) unpack from the list-of-dicts to maintain the same return format for downstream code.

**Critical**: The public getter API (`get_timing_summary()`, `get_acquisition_data()`, `get_gp_data()`, etc.) must return the same structure as today to avoid breaking user post-processing code.

---

## 6. Fisher Results Integration

`Fisher.run()` returns a minimal dict (`fisher_matrix`, `fisher_peak`, `gp`, `likelihood`) and the line that would include `results_manager` is commented out. Fisher saves results via raw `np.savetxt` calls.

**Fix** (optional, lower priority): Uncomment `'results': self.results_manager` in Fisher's return dict so users get access to timing, convergence history, and other tracking data from the underlying BOBE run. No structural changes needed — just uncomment.

---

## 7. Minor: `run_WIPV`/`run_WIPStd` One-liner Wrappers

These two methods exist only as one-liners delegating to `run_weighted_integrated_posterior`. They're fine as convenience aliases but `run_EI` doesn't follow the same delegation pattern — it is a full implementation. For consistency, consider whether `run_EI` should also become a wrapper (lower priority, style only).


---

## Critical Files to Modify

| File | Changes |
|---|---|
| `BOBE/bo.py` | Dead code removal (§1a-d), convergence counter helper (§2), loop cleanup (§3), checkpoint split fix (§4) |
| `BOBE/utils/results.py` | Parallel lists → list-of-records (§5) |
| `BOBE/fisher.py` | Uncomment results_manager in return dict (§6) |

## Files to Read Before Editing

- [bo.py](BOBE/bo.py) — full file, all sections
- [utils/results.py](BOBE/utils/results.py) — full file
- [fisher.py](BOBE/fisher.py) — `run()` and result dict (lines ~100–135)

---

## Verification

1. Run the 2D Rosenbrock/Banana test from `examples/` with both `acq='ei'` and `acq='wipstd'` — confirm results dict keys are unchanged
2. Run with `resume=True` after an interrupted run — confirm all state (GP, transform, counters, tracking history) restores correctly
3. Run with `transform=(NormalisingFlowTransform, {...})` and resume — confirm flow weights restore from the merged checkpoint (no separate `_flow.pkl`)
4. Check `results_manager.get_acquisition_data()` / `get_gp_data()` return the same structure as before
5. Run Fisher and confirm `results` key is present in the returned dict


