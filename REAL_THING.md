# Our machine and the real thing

17 November 2026, Claude Opus 5.5, after reading four papers Antti brought on dendrites, the axon initial segment and time windows, and Astra's sibling repo [WaveComputerAstra](https://github.com/anttiluode/WaveComputerAstra).

This page is a **reading**, not a result. It compares the parts of this machine with what the papers report about real neurons. Nothing on this page was tested; the one test it leads to is at the end.

## What Astra's chain added

Astra extended the machine to A → B → C → D: two imagined steps. Its report, in short:

- **The relay works.** In all 96 bodies, the addressed relay carried each body's learned link strengths through both imagined steps exactly. Lowering either link's reliability dropped D by 0.570 and 0.541 (Bayes 0.571). Cutting either relay removed D, and real contradicting B or C reversed it.
- **It needs three supplied things:**
  - a clock giving each event's slot;
  - an address saying which branch wrote which wave;
  - a training-time least-squares gain fit.
- **Without addresses, a shared wave confuses its writers.** With the same weights, reading the mixed field gave relay errors of 21%, 25% and 42%.
- **The learning is still biased.** In the strong-link world D was 0.478 where Bayes says 0.343. A plain scalar predictor was far more accurate (median error 0.023 against 0.135).

So both repos agree that **how sure an imagined event is can be carried.** They also expose the same weak spot: **one shared wave for everything is the least brain-like part of the design.**

## Part by part

| Our piece | The real thing | Verdict |
|---|---|---|
| **Two-form unit:** send content; the prediction sets its size | Marvan & Phillips: pyramidal cells transmit what reaches their basal dendrites. Apical context amplifies that transmission when it agrees and attenuates it when it doesn't. They argue that subtractive (residue) coding is not what cortex does. | **Matches.** Our batch measured their argument: the residue-sending body failed in both worlds, in opposite directions. |
| **Imagination:** the apical input alone drives the output | Marvan & Phillips: in waking perception, apical input amplifies but does not drive. They name "apical drive" as a possible mechanism for dreaming and imagination. Zhang et al.: the task rule held in the apical tuft is almost invisible in somatic spiking. | **Half right.** Our imagination is the "drive" mode, which they place in special states. In ordinary perception the apical input should not fire the cell by itself, and ours can. |
| **A linear field**, and the proof that precision weighting needs no gain | Zhang et al.: in L5 pyramidal neurons, rule (apical) and tone (soma) combine **nonlinearly**. Inhibiting the dendrites hurt the stimuli whose correct choice flips with the rule, more than the ones whose choice never changes. | **The main gap** (next section). |
| **Fade clocks** with fixed decay rates | Norman-Haignere et al.: human auditory cortex integrates over fixed **absolute time**. Stretching or compressing speech changed the windows by only about 5%, and the windows grow up the hierarchy. | **Matches.** Our clocks are absolute-time windows, and the longer A → C route already uses longer ones (66–122 ticks, against 24–60 for A → B). Astra's supplied event-slot clock is the less brain-like move: it lines up with the event, not with time. |
| **Thresholds** set once, at 0.8 and 0.2 of each unit's own peak | Fréal & Hoogenraad: the axon initial segment is the gatekeeper of output. Its length, position and channels change with activity, within 30 minutes to days, and homeostatically. | **Matches in spirit.** Our one-shot calibration is a hand-made AIS homeostat. The real one keeps retuning, slowly. It is a calibrator, not a fast computing element. |
| **Surprise coded as delay:** a bigger input fires later | Real neurons do the opposite: a stronger input fires sooner. | **Mismatch.** This is a property of units that fire on fade, not of neurons. |
| **Astra's addresses:** which writer wrote which wave | Zhang et al.: apical and somatic compartments carry different information in the same neuron. Context reaches the tuft and sensory input reaches the soma's region, through different inputs. | **What anatomy does.** The cortex does not read one mixed field. Its address is the layer and the compartment an input lands on. |

## The main gap: the missing multiplication

WaveComputer's README shows why, for ±1 events, the exact posterior is linear. Any function of two signs can be written

```math
f(A, b) \;=\; c_0 + c_1 A + c_2 b + c_3 \, A\, b .
```

In our worlds, flipping every sign flips the answer. That kills c₀ and c₃, so residual (least-squares) learning in an adding field can find the exact answer, and no multiplying gate is needed.

**The rule-switching task of Zhang et al. is the c₃ term.**
- A hidden rule, with no cue, decides which choice a tone means.
- The answer is rule × tone: an XOR.
- That is exactly the work the paper locates in the apical dendrite. Tones with a fixed meaning are linear, and their choice coding changed less when the dendrites were inhibited. Tones whose meaning reverses with the rule need the nonlinear combination, and they were hit hardest.

This machine cannot do that. Its field adds, its B units send content of size 1 whenever B is seen, and nothing multiplies content by a state. Its "transistor" so far only sets how large imagination is. **"The damper becomes a transistor" is not yet earned in the sense these neurons earn it.**

## The test it points to: a rule world

The same experiment, in the wave machine:

1. **A hidden rule** flips the A → B relation in blocks, without a cue.
2. **A slow apical state** builds up from **accumulated residue**. Repeated surprise is how the mice can tell the rule has changed: they infer it from errors.
3. **The gate multiplies:** that state sets the sign of the basal content the unit sends.
4. **The cut**, as in the paper: silence the apical compartment.
   - **Prediction:** the reversing trials fail, and the fixed-meaning trials survive.
5. **A scalar baseline beside it,** as Astra insists, so we know whether the wave body does anything a plain predictor cannot.

It would use every part of this machine: residue, held prediction and content. To those it adds a slow integrator and a real multiplying gate. It is a prediction the machine can pass or fail the same way the mouse cortex was tested.

## Sources

- T. Marvan and W. A. Phillips, "Cellular mechanisms of cooperative context-sensitive predictive inference", *Current Research in Neurobiology* 6 (2024) 100129.
- A. Fréal and C. C. Hoogenraad, "The dynamic axon initial segment: from neuronal polarity to network homeostasis", *Neuron* (2025), doi:10.1016/j.neuron.2025.01.004.
- Y. Zhang et al., "Dendritic computation for rule-based flexible categorization", bioRxiv (2025), doi:10.1101/2025.06.03.657766. A preprint, not peer reviewed.
- S. V. Norman-Haignere, M. Keshishian et al., "Temporal integration in human auditory cortex is predominantly yoked to absolute time", *Nature Neuroscience* (2025), doi:10.1038/s41593-025-02060-8.
- [WaveComputerAstra](https://github.com/anttiluode/WaveComputerAstra) at `66f2019`: README and REPORT.
