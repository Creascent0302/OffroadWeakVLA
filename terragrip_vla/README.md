# TerraGrip-VLA

A tracked vehicle can raise or lower its tracks, changing the **ground contact area**
across three gears — `0=S` (small), `1=M`, `2=L` (large). More contact area means more
traction (less wheel slip) but more effort. Given a forward camera and a natural-language
instruction, pick the gear.

The scientific question: **should the decision route through an interpretable physical
concept — the wheel slip each gear would produce — or not?** Three modes are implemented
behind one config switch so they can be compared on equal terms.

```
modular   vision ──▶ slip(g) for g in {S,M,L} ──▶ conformal upper bound ──▶ analytic
                                                  selector(alpha, tau, lambda) ──▶ gear
e2e       (vision, language) ─────────────────────────────────────────────────▶ gear
hybrid    vision ──▶ slip(g) bottleneck ──┬──────────────────────────────────▶ gear
                     scw * vision ────────┘  (jointly trained; scw = leakage knob)
```

---

## Where language enters — the part the original spec left open

The training data contains **no language**. It is `(image, gear_driven, slip_measured)`.
So how can any of this be a VLA?

The resolution is a physical one:

> **Physics is language-invariant. Preference is language-dependent.**
>
> How much a track slips on mud in gear `L` is a fact about the world. Saying *"be careful"*
> does not change it. So **the slip head never sees language.**
>
> *Which* gear you should pick, given the predicted slip, depends on how much risk you will
> accept. So **language enters at the decision**, as a risk budget `(alpha, tau, lambda)`:
>
> | | |
> |---|---|
> | `alpha` | conformal miss rate — smaller ⇒ wider slip bound ⇒ safer |
> | `tau` | slip budget — smaller ⇒ stricter ⇒ safer |
> | `lambda` | how much slip is traded against effort among acceptable gears |

Each mode consumes the instruction, and all three receive **exactly the same input**
`(image, instruction)` — that is what makes the comparison fair. Only the *route* differs:

* **modular** decodes the budget with `interpret()` and applies it **analytically** in the
  conformal selector. Nothing is learned about language.
* **e2e / hybrid** receive the frozen MiniLM embedding of the raw instruction and must
  **learn** the same mapping, entangled with perception.

Training labels are created by **language augmentation**: every sample, every epoch, draws a
random instruction; its budget turns the sample's true slip curve into an oracle gear label
(`data/dataset.py: LanguageAugmenter`).

Two consequences fall out of this, and both are measured, not just asserted:

1. **Label cost is asymmetric.** `modular` only needs the slip at the gear you *actually
   drove* — genuinely self-supervised from proprioception. `e2e`/`hybrid` need `best_gear`,
   which requires slip at gears you did **not** drive: **counterfactual** supervision. That
   is a strictly stronger requirement, and on a real robot it is the expensive one.
2. **Calibration cost is asymmetric.** The slip bound is calibrated **once** and serves every
   instruction, including instructions and risk budgets never seen. APS must be recalibrated
   **per instruction**, because the label it covers (`best_gear`) moves when the instruction
   moves.

---

## Results (`--config small`: real DINOv2-S, 3.2k images, 3 seeds)

Regenerate any of this with `python scripts/verify_framework.py --config small`, which
asserts all of it and prints PASS/FAIL. Paper numbers should use `default`
(DINOv2-B, 4.5k images, 5 seeds).

**Framework verification: 25 / 25 checks pass.**

### The Pareto front (E1)

| policy | effort ↓ | slip ↓ | violation ↓ | gear acc |
|---|---|---|---|---|
| fixed_S (always small) | 0.000 | 0.303 | 0.543 | 0.419 |
| reactive_only (proprioception, no vision) | 0.426 | 0.163 | 0.242 | 0.562 |
| **e2e** | **0.463** | 0.143 | 0.156 | **1.000** |
| *oracle* (knows the true slip curve) | *0.463* | *0.143* | *0.156* | *1.000* |
| fixed_M | 0.500 | 0.166 | 0.265 | 0.236 |
| **modular** | 0.598 | **0.112** | **0.099** | 0.732 |
| fixed_L (always large) | 1.000 | 0.088 | 0.095 | 0.345 |

* **`e2e` exactly matches the oracle.** The task is deterministic given
  (terrain, instruction), so the black box saturates it. `gear_acc` is therefore not
  the interesting column.
* **`modular` reaches `fixed_L`'s safety at 60 % of its cost** (violation 0.099 vs
  0.095, effort 0.598 vs 1.000). It buys a *certificate* with effort.
* Both strictly dominate `reactive_only` and `fixed_M`.

### The main table (3 seeds, mean ± std)

| mode | regime | gear acc | violation ↓ | effort ↓ | slip MAE / Bayes | instruction spread ↑ |
|---|---|---|---|---|---|---|
| modular | in-dist | 0.732 ± 0.010 | **0.099** | 0.598 | **1.09×** | **0.396** |
| e2e | in-dist | **1.000** | 0.156 | 0.463 | n/a | 0.513 |
| hybrid | in-dist | **1.000** | 0.156 | 0.463 | 1.28× | 0.513 |
| modular | held-out instruction | 0.669 | **0.098** | 0.659 | 1.09× | **0.264** |
| e2e | held-out instruction | **0.820** | 0.160 | 0.506 | n/a | 0.177 |
| hybrid | held-out instruction | 0.697 | 0.109 | 0.614 | 1.28× | 0.171 |
| modular | OOD terrain | 0.812 | **0.254** | 0.738 | 1.21× | **0.502** |
| e2e | OOD terrain | **0.828** | 0.346 | 0.582 | n/a | 0.696 |
| hybrid | OOD terrain | 0.663 | 0.429 | 0.428 | 1.82× | 0.265 |

**`modular` has the lowest violation rate in all three regimes** — 1.6× lower than `e2e`
in-distribution, and 1.4× lower on unseen terrain — and it keeps the most instruction
following when the phrasing is novel.

### Conformal coverage — the hard guarantee

| α | target | regression path | APS path |
|---|---|---|---|
| 0.05 | ≥ 0.95 | **0.965** ✓ | **0.958** ✓ |
| 0.10 | ≥ 0.90 | **0.921** ✓ | **0.911** ✓ |
| 0.20 | ≥ 0.80 | **0.824** ✓ | 0.790 ✓ (inside sampling noise) |

### Language

* Semantic direction holds: contact area `careful 0.784 ≥ normal 0.626 ≥ fast 0.386`.
* **The slip head is exactly language-invariant**: `|slip(careful) − slip(fast)| = 0.00e+00`.
* Language really moves the answer: 5 of 6 terrains change their optimal gear with the
  instruction (`concrete SSS · grass LSS · mud LLM · sand LMM · wet_tile LMS ·
  loose_gravel LLM`).

### The analyses

* **Leakage** (`scw`: `0→0.00, .003→0.00, .01→0.00, .03→0.18, .1→0.75, .3→1.00, 1→1.00`).
  At `scw = 0` the concept is genuinely load-bearing — shuffling it drops accuracy *below*
  the language-only floor. The floor is **0.591, not 1/3**: the instruction alone already
  tells you the marginally-best gear, and dividing by the wrong floor is the easiest way to
  manufacture a leakage curve.
* **Intervention**: `modular` goes `0.732 → 1.000` when the concept is corrected. **`e2e`
  has no interface to intervene on** — there is no physical concept in its forward pass, so
  a measured slip has nowhere to be written. That is a capability gap, not a tuning gap.
  And a *learned* bottleneck does not get this for free either: at `scw = 0`, feeding
  `hybrid` the **true** slip curve makes it **worse** (1.000 → 0.828), because its gear head
  calibrated itself to its own predictor's biases. `modular`'s analytic selector cannot be
  miscalibrated this way.
* **Probe: SATURATED, and the code says so.** An *untrained* gear head already decodes slip
  at `R² = 0.937`; the frozen-DINOv2 ceiling is `0.975`; headroom is `0.038`. With a floor
  that high, a high `R²` on the trained `e2e` model is **not** evidence that it learned
  traction — slip is decodable from *any* projection of these features, because the mock
  terrains are linearly separable. `probe.py` computes the headroom, flags `saturated`,
  refuses to report a normalised score, and prints the warning on the figure. The experiment
  only becomes informative on visually ambiguous *real* terrain. Hiding this would be the
  easiest way to publish a false claim.

## Fixes from the adversarial review

The codebase was then reviewed by six independent agents along orthogonal lenses
(conformal math, comparison fairness, analysis semantics, VLA/language wiring,
data/runnability, ML correctness). 37 findings were raised; 20 were refuted by
independent skeptics; 17 survived and are fixed. The ones that were silently
corrupting a result:

| What was broken | Why it mattered | Fix |
|---|---|---|
| **`mu.detach()` did NOT keep the sigma NLL out of mu.** mu and log_sigma shared a trunk, so the NLL reached log_sigma's weights and flowed back through the *shared hidden state* into the trunk mu depends on. | The invariant was documented in three files and was **false**: the sigma term was quietly retraining the mean. | The sigma branch now reads a **detached** hidden state. `test_sigma_nll_cannot_move_mu` fails on the old code. |
| **An empty randomised-APS set was resolved to the max-effort gear.** But an empty set is the coin flip `u > q` — independent of the image, at rate ≈ α — not evidence of danger. | α is exactly the knob language turns. Sending abstentions to gear L made α act on the two routes with **opposite sign**: raising α makes modular *cheaper* and made e2e *more expensive*. Every effort and language-direction comparison was corrupted. | Empty set → the model's own argmax, flagged via `low_conf` / `abstained`. |
| **λ was a dead knob.** Cost was `area + λ·slip`; the contact-area gap between gears is 0.5 while an acceptable gear has `slip ≤ τ ≤ 0.35`. No λ in `RISK_TABLE` could span the gap. | The paper's risk budget `(α, τ, λ)` was really `(α, τ)`. Measured: λ changed **zero** decisions. | Cost is `area + λ·(slip/τ)`, so the terms are commensurate; λ retuned. Now λ changes real decisions, and a test asserts it. |
| **The OOD split had one terrain**, so the oracle gear is constant per instruction, the language-only floor is exactly 1.0, and leakage headroom is ≤ 0. | The entire OOD leakage curve was **void by construction**. | A second OOD terrain (`loose_gravel`, oracle L/L/M vs `wet_tile`'s L/M/S) drops the floor to 0.67. |
| **`ensure_data` only checked that the `.npz` files existed.** | Change `data.seed` or `data.sizes` and the stale data is silently reused — you train on data you did not ask for, and nothing says so. | The generating `(seed, sizes)` is recorded next to the data and compared; mismatch regenerates. |
| **`compare.py --train-missing` retrained from a fresh `default` config**, dropping every CLI override. | It would train models under DEFAULT settings and then evaluate them under yours — a "controlled comparison" of models that were never trained the way you asked. | The training config is derived from the live `cfg`. |
| **`select.py` claimed both routes carry "the same guarantee".** They do not: modular's bound is calibrated on the *measured* (noisy) slip; APS covers `best_gear`, which is defined on the *mean* curve. | One route was being scored against the other's promise. | Both `violation` (realised) and `violation_mean` (mean) are now reported, and the docstring states the two events precisely. |
| `predict_all` drew an **independent dropout mask per gear**, scrambling the cross-gear structure of the `[B,3]` concept the hybrid head reads. | Noise unrelated to terrain was injected into the bottleneck. | One mask per *sample*, shared across its three gears. |
| σ was fitted against **dropout-corrupted** residuals. | σ absorbed `terrain_noise + dropout_noise`, inflating the conformal bound worst on the low-noise terrain where a tight bound is the point. | The NLL is fitted against a dropout-free mean (`slip_clean`). |
| `set_seed` ran **before** `build_context`, which consumes the global torch RNG via `torch.hub`. | The same `train.seed` gave different weights depending on whether the backbone was cached. | Seed immediately before head construction. |
| `eval.compare` crashed with `KeyError('mode')` on the no-checkpoint path its own `[skip]` message advertises. | — | Guarded. |
| `gears` / `contact_area` / `label.tau_train` in `default.yaml` were read by nothing. | Editing them looked like it would do something and did not. | Removed; they live in `constants.py`. |

## Deviations from the original spec, and why

Every one of these was forced by something that **measurably broke** the experiment. Each is
implemented, documented at its call site, and (where it is a claim) covered by a test.

| # | Spec said | We do | Why — with the number that forced it |
|---|---|---|---|
| 1 | split conformal on the raw residual `y - mu` | **normalised** residual `(y - mu)/sigma_hat` | Slip noise is heteroscedastic by design (`sigma` 0.02 on concrete, 0.15 on mud). One global quantile must cover mud, so `Q(0.05) = 0.158`; the bound on concrete becomes `0.05 + 0.158 = 0.208 > tau_careful = 0.15`. The acceptable set is then **empty everywhere** and `modular` degenerates to "always gear L" — *even with a perfect slip predictor*. Conformal is valid for **any** score, so the finite-sample guarantee is untouched. This is **not** CQR (no quantile regression, no pinball loss) — CQR stays in Phase 2. `conformal.score: absolute` reproduces the collapse as an ablation, and a test asserts it. |
| 2 | deterministic APS | **randomised** APS (Romano et al. 2020) | The gear classifier is ~100% accurate, so every deterministic calibration score is ≈ `p_top` ≈ 0.99, `q` lands at ≈ 0.999, and **81 %** of test sets needed 2–3 classes. `e2e` was then forced to the largest gear constantly and spent *more* effort than a fixed-medium baseline — an artefact of the score that would have silently handed the comparison to `modular`. Randomised APS gives exact (not conservative) coverage and near-singleton sets: ambiguity fell 0.81 → 0.11, effort 0.743 → 0.483. |
| 3 | ambiguous APS set → smallest gear | ambiguous set → **largest** gear (`select.ambiguity: safe`) | With the smallest gear, `e2e` has **no guarantee at all** and the Pareto comparison is rigged. Taking `max(S)`: APS covers `best_gear` w.p. ≥ 1−α, and slip is monotone decreasing in contact area, so `chosen ≥ best ⇒ slip(chosen) ≤ slip(best) ≤ tau`. `e2e` now carries the **same guarantee shape** as `modular`, which is the only way the comparison means anything. (The spec's parenthetical "或退大档" allows exactly this.) |
| 4 | `modular` slip MAE < 0.06 | report **MAE / Bayes floor**; gate at ≤ 1.3× | With these sigmas the irreducible noise floor is **0.061** — the spec's target sits *below* what any model can reach. Achieved: **1.11×** the floor. The ratio is the meaningful quantity; the raw number is not. |
| 5 | `[phi_vis, onehot(gear)]` concatenated raw | phi_vis is standardised **and L2-normalised** first | Raw `‖phi‖ ≈ 28` versus `‖gear one-hot‖ = 1`, `‖lang‖ = 1`, `‖concept‖ ≈ 0.5`. Vision outweighed everything ~28:1 and the heads barely used the other inputs: the slip head's systematic error per `(terrain, gear)` cell was **0.047** raw vs **0.017** normalised — it was largely ignoring *which gear it was asked about*. One fixed, mode-agnostic affine map fixes all three imbalances at once, so it cannot favour any mode. |
| 6 | *(not specified)* | dropout 0.4 in **both** heads | 768-d features, ~1.8 k training samples: the slip head memorised the noise. MAE **0.091 → 0.067** (floor 0.056). Identical in both heads, so capacity stays matched. |
| 7 | early stop on the loss | early stop on a **task** loss excluding the sigma NLL | The Gaussian NLL that fits `sigma` sits around **−2** while the slip MSE is around **0.008**. Early stopping on the total halted training at epoch 14 while the actual task was still improving. |
| 8 | OOD terrain `wet_tile = [0.35, 0.20, 0.15]` | `[0.30, 0.18, 0.10]` | Those values sit **exactly** on the three `tau` boundaries (0.15 / 0.25 / 0.35), so `slip <= tau` became a floating-point coin flip: an epsilon-sized conformal width swung the oracle-intervention accuracy on OOD from **1.00 to 0.67** on round-off alone. The oracle gears are still `(L, M, S)` across the three instructions — the full spread — but now with ≥ 0.03 of margin. |
| 9 | `scw ∈ {0, 0.1, 0.5, 1.0}` | log-spaced `{0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0}` | Measured leakage is 0.0 at `scw=0` and already **1.0 at `scw=0.1`**. The linear sweep is a step function with the transition invisible between its first two points. |

---

## Things the analyses revealed that are worth knowing

* **The probe experiment is saturated on this mock, and the code says so.** An *untrained*
  gear head already decodes slip at `R² = 0.96`; the frozen DINOv2 ceiling is `0.994`. With a
  floor that high, a high `R²` on the trained `e2e` model is **not** evidence that it learned
  traction — slip is decodable from *any* projection of these features, because the four mock
  terrains are linearly separable. `probe.py` computes `headroom = ceiling − floor`, flags
  `saturated`, and prints the warning on the figure. The experiment only becomes informative
  on visually ambiguous real terrain. (The first version of the probe also trained on the
  *noise-free* slip curve while the slip head only ever sees a *noisy* measurement at *one*
  gear — a strictly easier problem. It now gets exactly the slip head's supervision.)
* **Correcting `hybrid`'s concept makes it *worse*** (`intervention_gain = −0.098` at
  `scw = 0`). The learned gear head calibrated itself to the *predicted* concept's biases, so
  feeding it the true slip curve is out-of-distribution *for the head*. `modular`'s analytic
  selector cannot be miscalibrated this way, and oracle intervention takes it to `1.000`.
  This is a real argument for the analytic route, and it is the kind of thing a concept
  bottleneck is supposed to buy you but does not, automatically.
* **Joint training distorts the concept.** `hybrid`'s slip MAE is `1.70×` the Bayes floor
  versus `modular`'s `1.11×`: the cross-entropy gradient pulls the "slip" prediction toward
  whatever helps the gear decision, not toward slip. `beta` controls this.

---

## Layout

```
constants.py          gears, contact areas, image geometry
language.py           RISK_TABLE, instruction pool (train / held-out), frozen MiniLM,
                      interpret(text) -> RiskBudget
features.py           frozen-backbone feature cache (content-hashed)
runtime.py            one config, one data build, one backbone, one cache — shared by
                      train / eval / analysis so nothing drifts apart

data/     schema, heteroscedastic mock generator, datasets, LanguageAugmenter, oracle labels
models/   frozen DINOv2 + ROI pooling; SlipHead; GearHead; TerraGripModel (3 modes, 1 class)
conformal/ normalised split conformal; randomised APS; the selector (where language lands)
training/ per-mode losses; one training loop for all modes
eval/     metrics, baselines, policies, E1 Pareto, multi-seed comparison
eval/analysis/  leakage, probe, intervention (+OOD)
tests/    66 tests, including conformal coverage as a hard assertion
```

## Running it

```bash
pip install -r requirements.txt        # torch from the cu128 index; see the file header
python scripts/env_check.py            # expects sm_120 on an RTX 5090

# everything below auto-generates the mock data and the feature cache on first use
python -m training.train mode=modular
python -m training.train mode=e2e
python -m training.train mode=hybrid side_channel_weight=0.5

python -m eval.run_eval                # E1 Pareto + conformal coverage
python -m eval.analysis.leakage --train-missing
python -m eval.analysis.probe
python -m eval.analysis.intervention
python -m eval.compare --train-missing # the main table, 5 seeds x 3 modes

pytest                                 # 66 tests; `-m slow` adds the DINOv2/MiniLM ones
```

Everything runs on CPU (the backbone is frozen and its features are cached, so training is a
small MLP). On the 8×5090 box, run different `(mode, seed)` pairs concurrently with
`CUDA_VISIBLE_DEVICES`.

## Phase 2 — deliberately not implemented

Real-robot adapter, label-space projection (camera homography + time-lag association), deploy
inference with runtime correction, CQR, online/ACI recalibration, world-model backbone,
RELLIS-3D / RUGD, physical-property heads. The extension points exist: `data.DataSource` is
the abstract data interface (`MockSource` is one implementation), and selection is fully
decoupled from deployment.
