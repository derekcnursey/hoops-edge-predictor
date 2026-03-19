# Neutral-Site Symmetry Handoff

## Current State / Recommendation

- A real neutral-site slot bias existed in the trained `HGBR` mean model.
- Production/live neutral-site inference is now explicitly symmetrized for:
  - `mu`
  - `home_win_prob`
  - `sigma`
- Canonical walk-forward / historical site archive was also updated so neutral-site holdout predictions are symmetrized.
- `DCN INDEX` was already safe by construction because it used both slot orientations.
- Do **not** adopt swapped-slot augmentation for the mean model.
- Current recommendation:
  - keep inference-time neutral-site symmetrization
  - keep swapped-slot augmentation as dropped research

## What Was Tested

### 1. Neutral-site bias audit
- Historical neutral-site swap tests showed strong nominal home-slot bias in the raw trained model.
- Baseline pooled raw neutral-game slot-bias:
  - mean slot bias: `+3.10`
  - mean absolute slot bias: `3.56`
  - p95 absolute slot bias: `8.69`
  - home-slot advantage share: `84.97%`

### 2. Immediate production fix
- Implemented explicit neutral-site inference symmetrization in live prediction.
- Symmetrized:
  - margin / `mu`
  - win probability
  - sigma
- This protects March/tournament predictions now without retraining.

### 3. Historical / benchmark fix
- Canonical walk-forward benchmark and historical archive were rebuilt so neutral-site holdout predictions are also symmetrized.

### 4. Training-side augmentation research
- Neutral-only swapped-slot augmentation under the old contract:
  - reduced slot bias
  - did not earn promotion on benchmark quality
- `swap_safe_v2` feature contract:
  - added as research-only
  - benchmark drift was exactly zero
- Neutral-site-only swapped-slot augmentation on `swap_safe_v2`:
  - materially reduced intrinsic slot bias again
  - stayed closer to the benchmark than full all-game augmentation
  - still did **not** earn promotion
- Full all-game swapped-slot augmentation on `swap_safe_v2`:
  - materially reduced intrinsic slot bias
  - materially hurt the trusted benchmark, including the `Dec 15+` betting window

## What Won / What Lost

### Won
- **Inference-time neutral-site symmetrization**
  - fixes the highest-priority March risk
  - keeps non-neutral behavior unchanged
- **swap_safe_v2 contract**
  - safe research foundation
  - zero benchmark drift by itself

### Lost
- **Neutral-only swapped-slot augmentation**
  - not strong enough to promote
- **Neutral-site-only swapped-slot augmentation on `swap_safe_v2`**
  - reduced bias materially
  - benchmark impact was small but still not clearly positive
  - not strong enough to promote
- **Full all-game swapped-slot augmentation on swap_safe_v2**
  - strong bias reduction
  - unacceptable benchmark damage

Key neutral-only `swap_safe_v2` result:
- pooled `MAE_all`: `9.5984 -> 9.6095`
- pooled `MAE_lined`: `9.0971 -> 9.0964`
- `Dec 15+ MAE_all`: `10.2759 -> 10.2882`
- `Dec 15+ MAE_lined`: `9.3765 -> 9.3670`

Key neutral-only `swap_safe_v2` bias result:
- mean slot bias: `+3.10 -> +1.14`
- mean absolute slot bias: `3.56 -> 1.94`
- p95 absolute slot bias: `8.69 -> 5.09`
- home-slot advantage share: `84.97% -> 67.76%`

Interpretation:
- neutral-only augmentation under the clean contract is directionally better on symmetry
- but still not strong enough to replace the current baseline mean model
- it remains research-only

Key full all-game augmentation result:
- pooled `MAE_all`: `9.5984 -> 9.8564`
- pooled `MAE_lined`: `9.0971 -> 9.2729`
- `Dec 15+ MAE_lined`: `9.3765 -> 9.5467`

Key bias reduction result from full all-game augmentation:
- mean slot bias: `+3.10 -> +0.06`
- mean absolute slot bias: `3.56 -> 1.04`
- p95 absolute slot bias: `8.69 -> 2.69`
- home-slot advantage share: `84.97% -> 52.12%`

Interpretation:
- the augmentation solved the symmetry problem mechanically
- but hurt the actual predictive benchmark too much

## Do Next / Do Not Do Next

### Do next
- Keep the current live neutral-site symmetrization in place.
- Treat `swap_safe_v2` as the correct research contract if symmetry work is revisited later.
- If this branch is revisited, test only more targeted symmetry approaches that preserve benchmark quality.

### Do not do next
- Do **not** make swapped-slot augmentation the default mean-model training path.
- Do **not** remove the live neutral-site symmetrization fix.
- Do **not** assume model-level neutrality is solved just because inference is fixed.
- Do **not** spend more time on brute-force all-game slot duplication unless a new representation or regularization idea justifies it.
- Do **not** spend more time on neutral-only slot duplication either unless a new approach shows a clearer benchmark path.
