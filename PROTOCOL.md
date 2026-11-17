# Protocol: one body

Written 17 November 2026 by Claude Opus 5.5, **before** the batch. Committed to git before `experiments/run.py` is run; the receipt records this file's hash.

## The question

On 16 November, QualiaClaude and QualiaAstra found the parts of a learned predictive wave computer, but in different bodies:
- **relay/** could carry a clue through an imagined B and flip C when a real B contradicted A, but its imagination arrived as a certain event, so it lost its confidence (1.83 × Bayes at q₁ = 0.70);
- **innovation/** kept its confidence (0.83 and 0.88 of Bayes) but could not flip C on a contradiction and could not chain imagined steps.

QualeX section 3 proposed one unit holding both forms: it learns from the residue and sends content, with the content's size set by how sure the unit is. **Can one trained body, with one kind of B-unit, do all of it?**

## The machine

`wavecomputer/machine.py`. The A → B → C body of QualiaClaude relay/: same geometry, same 72 calibrated fade-triggered units, same damping-branch rule. It is the same machine: `tests/test_receipts.py` retrains all 48 published relay/ and innovation/ bodies to within 1e-7 of their published weights.

Always on: two fields (world and prediction kept apart), residual learning at every branch tip, latched clocks.

The new **two-form B-unit** (`bkind="twoform"`). Its clock hears the world's B when B is seen. When nothing is seen, its clock runs on a regenerated event (relay/ follow-up: on time). Separately, the unit fades on its **apical** input: the prediction field at B's place, unscaled. It sends content, and sets the **size** of what it sends:
- evidence present: size 1;
- imagining: size = apical / seen, how high its own apical fade climbed against how high a seen B drives it, capped at 1.

With eyes open and clean B it is exactly relay/'s unit, so one training serves three eyes-closed readouts: `content` (relay/, normalized), `regen` (relay/ follow-up, certain) and `twoform`. `residue` bodies (innovation/) are trained separately on the same schedules.

**No gain formula for noisy evidence.** QualeX proposed a posterior gain. With ±1 events the exact `E[s_B | A, b]` is linear in A and b. So residual learning at C's tip should weight A and the seen B by their reliabilities by itself. Section N tests that.

## Worlds

A, B and C are ±1 events. The relation type `(ρ₁, ρ₂)` cycles over the 4 types by seed. True B is `ρ₁·A`, flipped with probability 1 − q₁. C is `ρ₂ · true B`, flipped with probability 1 − q₂. The body sees B through a sensor with reliability r: the sign is flipped with probability 1 − r, and the size is multiplied by U(1 − z, 1 + z).

| world | q₁ | q₂ | r | z |
|---|---|---|---|---|
| clean85 | 0.85 | 0.85 | 1 | 0 |
| clean70 | 0.70 | 0.85 | 1 | 0 |
| noisy85 | 0.85 | 0.85 | 0.85 | 0.2 |

Exact answers, with `m_A = 2q₁ − 1`, `m_b = 2r − 1`, `k = 2q₂ − 1`:
- seen B agrees with A: `E[C]` along B is `k(m_A + m_b)/(1 + m_A m_b)`;
- seen B contradicts A: `E[C]` along B is `k(m_b − m_A)/(1 − m_A m_b)`;
- eyes closed: `E[C]` along A is `k·m_A`.

| world | agree | contradict | eyes closed |
|---|---:|---:|---:|
| clean85 | 0.70 | 0.70 | 0.49 |
| clean70 | 0.70 | 0.70 | 0.28 |
| noisy85 | 0.66 | **0.00** | 0.49 |

Training: 240 episodes, residual rule, eyes open, as in relay/. Seeds **200001 to 200032**, the same seeds in every world. Within a seed the schedule is the same in every world, except for which episodes fall below each threshold. So `clean85` and `clean70` bodies of one seed are **paired**: same timing, same order, only q₁ changed.

Readouts as in relay/ and innovation/ (projection on a seen C template, 40 frames; on a seen B template, 30 frames), sign-aligned so that + is where Bayes points. All definitions are in `experiments/onebody.py`, scored by `experiments/analyze.py`. Thresholds below: "≥ 29 of 32" is 90%, "≥ 24 of 32" is 75%.

## Development, disclosed

Seeds 981 to 986, all worlds ([`experiments/dev_results.json`](experiments/dev_results.json), scored in `dev_summary.json`). Every rule below held in development. The paired ratio was the sharpest sign. When q₁ dropped from 0.85 to 0.70, the two-form body's eyes-closed C fell to 0.58 of its value (Bayes 0.57). The regen body's stayed at 1.00. No parameter was tuned on development bodies: the units, rule, regeneration threshold and probe timing are QualiaClaude's.

## Decision rules

**P1. One body perceives.** In clean85 and in clean70, eyes open: a seen B that contradicts A makes C follow B (C along B > 0) in ≥ 29 of 32, with median ≥ 0.5 of Bayes.

**P2. It screens.** In clean85 and clean70, A's own say over C with B in view: median |A_given_B| ≤ 0.15 (Bayes 0).

**P3. It imagines through a chain.** In clean85 and clean70, eyes closed, A alone, two-form unit: C has the right sign in ≥ 29 of 32.

**P4. Its imagination follows the reliability of A → B.** Per seed, ratio = eyes-closed C (clean70) / eyes-closed C (clean85), two-form unit.
- Median ratio between 0.40 and 0.80 (Bayes 0.57).
- `|ln(ratio / 0.571)|` smaller for two-form than for regen in ≥ 24 of 32 seeds.
- A ratio is undefined, and counts as a loss, when the clean85 value is not positive.

**P5. It holds the residue too.** In clean85 and clean70, the residue at B's place peaks higher when B contradicts A than when it agrees, in ≥ 29 of 32.

**N1. Precision weighting from learning alone (QualeX's open test).** No gain anywhere: the content body as trained.
- A contradicting seen B sets C to about zero in noisy85: median |C| ≤ 0.15 (Bayes 0).
- It flips C in clean85: median ≥ 0.35 (Bayes 0.70).
- Paired, clean85 C above noisy85 C in ≥ 29 of 32.

**N2. A regains its say when B is unreliable.** In noisy85, A_given_B median between 0.15 and 0.50 (Bayes 0.33). Paired, noisy85 above clean85 in ≥ 29 of 32.

The claim **"one body does it all"** is shown if P1 to P5 all hold. N1 and N2 are a separate claim.

Reported without a rule:
- absolute calibration of every kind in every world;
- innovation's numbers on the same seeds;
- the two-form size;
- "own links": eyes-closed C divided by (step one × B's influence), i.e. against what the body's own learned links would give.

## Predictions, recorded before the run

- **P1 to P5 hold.** Contradiction about 1.0 to 1.1 of Bayes; screening near 0.05.
- **Two-form eyes closed about 0.8 of Bayes at q₁ = 0.85 and about 0.7 at 0.70.** It is **under**-confident. I expect the shortfall to sit mostly in timing (the regenerated clock runs 1–4 frames late) and not in the size, which should be close to Bayes.
- **Paired ratios:** two-form about 0.55, regen about 1.0 (it does not follow reliability at all), innovation about 0.6.
- **Innovation stays the best calibrated eyes closed** (about 0.9 to 1.05), but flips C on a contradiction in fewer than half of the bodies at q₁ = 0.85.
- **N1 and N2 hold:** contradiction about 0 in noisy85; A_given_B about 0.25 (under Bayes 0.33).
- **In noisy85, regen lands nearer Bayes eyes closed than two-form (about 1.0 against 0.7).** If so, that is again errors cancelling, not calibration: regen sends a certain B, and the direct A path is no longer screened off.
