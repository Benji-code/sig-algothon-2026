# Sigma34 Combined Live Alphas — Production Report

**Model:** `Sigma34_CombinedLiveAlphas.py`  
**Reported live score:** **943**  
**Status:** **Frozen primary fallback**  
**Successor:** Sigma42 Sigma34 + 10% Basket Only — 945 live  
**Date recorded:** 25 July 2026

---

## 1. Executive conclusion

Sigma34 Combined Live Alphas established the previous production record of **943 live** and became the validated chassis for subsequent research.

The model combines two independently live-positive structures:

1. **Sigma32 HighShrinkSparseSpectral** — a sparse-plus-spectral lead-lag estimator.
2. **Sigma25DG ComponentArbiter** — a stale-relationship death guard that flattens, preserves, or reroutes selected non-ALGO positions based on independent seed/gray and AG support.

Sigma34 does not introduce a new ALGO model. It retains the original five-day ALGO reversal and the established AG-shock vector.

Sigma42 later added a small basket-reversion sleeve to this exact chassis and scored 945. Sigma34 therefore remains the cleanest validated fallback and the correct baseline for attributing Sigma42's incremental gain.

---

## 2. Exact model identity

### File

`Sigma34_CombinedLiveAlphas.py`

### SHA-256

```text
e1dfdcf26ba1612314a5f8d4094686c59f7bca6abd4eea1af7e593bc095eff49
```

### Naming clarification

The exact code was originally stored as:

```text
Sigma33_SparseSpectralArbiter.py
```

After its live score of 943, the same model was promoted and referred to as:

```text
Sigma34 Combined Live Alphas
```

The packaged `Sigma34_CombinedLiveAlphas.py` is byte-for-byte identical to that original file. This production Sigma34 must not be confused with the later experimental files:

- `Sigma34_CommonTrendIdioReversal_Core.py`
- `Sigma34_CommonTrendIdioReversal_Arbiter.py`

Those ALGO-gate experiments scored 862 and 874 live and were rejected.

---

## 3. Model lineage

Sigma34 retains the following established components:

1. Seven frozen seed pairs.
2. Gray-tier pair relationships.
3. Two-factor residualisation:
   - remove contemporaneous ALGO exposure;
   - remove the largest remaining cross-sectional principal component.
4. Sparse-plus-spectral lead-lag estimation.
5. ALGO-shock conditional AG vector.
6. Original five-day ALGO reversal at the $100,000 ALGO limit.
7. Sigma25DG component-level stale-relationship arbitration.

No birth-catcher is active.

---

## 4. Sparse-plus-spectral lead-lag estimator

The Sigma32 core constructs expanding and recent lead-lag matrices from standardised residual returns.

### Expanding matrix

The expanding matrix uses:

```python
LL_FULL_SV_SCALE = 1.6
LL_FULL_SV_POWER = 0.7
```

It is decomposed into:

- a smoothly shrunk low-rank component; and
- a small sparse residual containing robustly extreme pair-specific entries.

The sparse correction restores selected strong relationships that would otherwise be erased by low-rank denoising.

### Recent matrix

The fixed recent matrix uses stronger shrinkage:

```python
LL_RECENT_SV_SCALE = 3.0
LL_RECENT_SV_POWER = 0.95
```

This reflects the higher estimation noise in the fixed 450-day recent sample.

### LL normalisation

The final LL signal uses a denominator blending:

- 90% of the AG-adjusted LL cross-sectional dispersion; and
- 10% of the LL-only dispersion.

This prevents the AG vector from fully determining the scale of the ordinary LL forecast.

---

## 5. Component-level death resolution

The DG overlay acts only after the base model has opened a non-ALGO position and the relationship-decay committee declares the underlying LL support stale.

It evaluates:

- total LL support in the current position direction;
- ordinary pre-AG LL support;
- recent-only LL contradiction;
- residual-level mean-reversion evidence;
- seed/gray support; and
- incremental AG support.

When a death flag is raised, the arbiter chooses among three actions:

### Flatten

Close the position when stale LL support lacks independent confirmation.

### Preserve

Keep the original position when either:

- seed/gray supplies strong independent support; or
- the AG-induced LL effect supplies strong independent support.

### Reroute

When seed/gray explicitly points the opposite way and neither preservation condition is met, route the position toward the seed/gray direction.

The overlay:

- never changes ALGO;
- never opens a position without a death flag;
- never enlarges a base position; and
- leaves the underlying sparse-spectral estimator unchanged.

---

## 6. Corrected historical validation

All results used the corrected official evaluator:

- 1 bp default commission;
- 0.2 bp ALGO commission;
- $10,000 default position limit;
- $100,000 ALGO position limit;
- fresh-process cold start for each 250-day test;
- official one-loop-lagged commission accounting;
- score \(=\mu SR^2/(SR^2+1)\).

| Model | Days 251–500 | Days 501–750 |
|---|---:|---:|
| Sigma25 spectral core | 754.1112 | 904.7417 |
| Sigma32 HighShrinkSparseSpectral | 755.7676 | 951.8319 |
| Sigma25DG ComponentArbiter | 768.3222 | 915.1220 |
| **Sigma34 Combined Live Alphas** | **769.9659** | **962.2155** |

Sigma34 therefore combines:

- Sigma32's stronger later-window estimator; and
- Sigma25DG's independently useful stale-position resolution.

---

## 7. Live evidence

| Model | Live score |
|---|---:|
| Sigma25DG ComponentArbiter | 929 |
| Sigma32 HighShrinkSparseSpectral | 933 |
| **Sigma34 Combined Live Alphas** | **943** |
| Sigma42 Sigma34 + 10% Basket Only | 945 |

Sigma34 improved the previous live-best Sigma32 by **10 score points**.

Its live result established that the DG overlay and sparse-spectral estimator were complementary in the hidden window.

---

## 8. Why Sigma34 remains important

Although Sigma42 now leads by two points, Sigma34 remains the key benchmark because:

- both of its major components have independent live support;
- it does not depend on the later basket sleeve;
- it is simpler than Sigma42;
- it is the exact chassis underlying Sigma42;
- it cleanly attributes Sigma42's incremental result to basket reversion; and
- it remains the safest fallback if the basket sleeve later proves unstable.

---

## 9. Rejected naming-collision branch

The later ALGO common-trend/idiosyncratic-reversal experiments also used the name Sigma34, but they are not this model.

Their live results were:

| Experimental model | Live score |
|---|---:|
| CommonTrendIdioReversal Core | 862 |
| CommonTrendIdioReversal Arbiter | 874 |

That ALGO gate was retired. It should never be merged into or confused with Sigma34 Combined Live Alphas.

---

## 10. Production decision

### Current leader

**Sigma42 Sigma34 + 10% Basket Only — 945 live**

### Frozen fallback

**Sigma34 Combined Live Alphas — 943 live**

### Frozen instruction

Do not alter Sigma34's:

- spectral shrinkage constants;
- sparse residual rule;
- recent LL shrinkage;
- AG threshold or weight;
- seed pairs;
- gray tier;
- DG decay committee;
- DG preservation/rerouting thresholds; or
- ALGO five-day reversal.

Any future research should preserve this file unchanged and use a new model number for every added mechanism.

---

## 11. Final verdict

Sigma34 Combined Live Alphas is the most important validated chassis in the model lineage.

Its live score of **943** demonstrated that:

> Sparse-plus-spectral lead-lag estimation and component-level stale-relationship arbitration add complementary live value.

Sigma42's 945 record is best understood as a narrow incremental improvement on top of this frozen model, not as a replacement of its core architecture.
