# AMT + LoRA Fine-Tuning — Observations & Experiments

**Date:** 2026-05-13
**Status:** Partial success — scoping confirmed for instruments, key, and durations; note density remains an open problem.

---

## 1. Background

**Goal.** Test whether a small LoRA adapter on the pretrained Anticipatory Music Transformer (AMT, `stanford-crfm/music-small-800k`) can scope the model to a single classical piano piece. Motivated by an earlier finding that the base model, when given a Mozart melody, scatters generated notes across 15+ instruments instead of producing piano accompaniment — a training-distribution mismatch (Lakh MIDI is mostly pop/rock).

**Test piece.** Brahms Hungarian Dance No. 5 (F# minor), from the ACCompanion dataset. Available as:
- `cc_solo/` — 9 recordings of primo (right hand) alone
- `cc_primo/` — 14 primo recordings
- `lb_secondo/` — 14 secondo (left hand) recordings
- `duet/` — 14 full-duet recordings

---

## 2. Pipeline

- **Base model:** `stanford-crfm/music-small-800k` (GPT-2 architecture, ~128M params)
- **LoRA config:** r=16, alpha=32, target `c_attn`, dropout 0.05, ~590k trainable params (0.46% of total)
- **Training:** AdamW, lr=2e-4, 100-step warmup, batch 8, sequence length 1024 tokens with stride 512
- **Hardware:** A100 (Colab)
- **Checkpointing:** Drive-mounted, every NUM_STEPS/20

---

## 3. Experiments

### Experiment 1 — Continuation mode, cc_solo data

Fine-tuned on 9 solo-piano primo recordings. Evaluated by prompting with 5s of one recording and asking for 10s autoregressive continuation.

**Results (single sample, 88 epochs, top_p=0.95):**

| Metric | Ground truth | Fine-tuned | Base |
|---|---|---|---|
| Note count | 71 | 50 | 176 |
| Instruments | [0] | [0] | [0, 3, 6, 18, 28, 41, 48, 50, 51, 66, 72, 75, 89, 111, 128] (15 total) |
| Pitch classes used | F# minor scale | F# minor scale | scattered |

Fine-tuned model committed to piano only and used exactly the F# minor scale, with zero notes on C, D#, G, A# — same as ground truth.

### Experiment 2 — Accompaniment mode, duet data

Fine-tuned on 14 duet recordings (both parts merged). Evaluated by feeding primo as anticipation controls + 5s of secondo as seed, asking model to generate the rest of secondo.

**Initial results (88 epochs, single-shot 15s window, top_p=0.95):**

| Metric | Ground truth | Fine-tuned | Base |
|---|---|---|---|
| Note count | 121 | 162 | 58 |
| Instruments | [0] | [0] | [0, 8, 26, 28, 40, 42, 48, 68, 76, 93, 102, 110] (12 total) |
| F# = 36, C# = 34 | matched | matched | wrong |

Looked like a clear win at first pass.

### Continuation of Experiment 2 — Professor's 's suggestion: more epochs, 20s output

Trained 500 epochs (later interrupted from the planned 3000-epoch run; not justified after diagnostics below). Generation switched to chunked 4 × 5s to cover 20 seconds.

**Results (500 epochs, 20s chunked, top_p=0.95):**

| Metric | Ground truth | Fine-tuned | Base |
|---|---|---|---|
| Note count | 170 | 53 | 104 |
| Pitch-class EMD | — | 0.29 (good) | 0.56 |
| Instrument Jaccard | — | 1.00 (perfect) | 0.07 |
| Mean duration diff (s) | — | 0.007 (essentially identical) | 1.281 |
| Mean IOI diff (s) | — | 0.091 | 0.043 |

Three of five metrics excellent. Note density at 31% of ground truth.

---

## 4. Diagnostic deep-dive (May 13)

Hypothesis: density regression caused by either (a) overtraining, (b) chunked generation thinning, (c) 20s window vs 15s window, or (d) sampling variance hiding a real limitation. Ran 6 step-by-step tests:

| Step | Setup | Result |
|---|---|---|
| 1 | List checkpoints on Drive | 15 checkpoints saved, all accessible |
| 2 | v1 (88 epochs) at 20s single-shot | 39 notes |
| 3 | v1 at 20s with `generate_resample` | 43 notes |
| 4 | v1 at 15s single-shot | 41 notes |
| 5 | v1 at 15s × 5 samples | 38, 39, 40, 40, 43 (mean 40, range 38–43) |
| 6 | v1 at 15s × top_p ∈ {0.92, 0.95, 0.98, 0.99} | All means in 40.3–42.3 range |

**Conclusion of diagnostic:** the model produces 40 ± 2 notes per 15-second window across every sampling configuration we tried. The earlier 162-note result was not reproducible — almost certainly a kernel-state artifact from a different sampler path or random seed.

---

## 5. Findings

**What the LoRA scoped successfully:**
1. **Instruments.** Base = 12–15 instruments. Fine-tuned = 1 (piano). Jaccard 1.00 vs 0.07.
2. **Pitch-class distribution.** Fine-tuned matches ground truth's F# minor scale. Pitch-class EMD 0.29 vs base 0.56. Notes outside the scale (C, D#, G, A#) correctly absent.
3. **Note durations.** Fine-tuned matches ground truth duration distribution to within 7 ms.

**What the LoRA did not scope:**
1. **Density.** Fine-tuned produces ~33% of ground-truth note count, consistently, across sample seeds and top_p settings.

**Interpretation.** The model has learned *what* to play (instrument, harmony, note length) but not *how often* to play it. With primo as anticipation controls, the model places sensible notes at sensible pitches with sensible durations — just not enough of them. The longer inter-onset intervals (mean IOI diff 0.091 s) confirm this: the model "skips beats" rather than playing arpeggiated fills the way the original secondo does.

**Negative result on training length.** More epochs did not help density. The 500-epoch model is no denser than the 88-epoch model on this metric. The literal 3000-epoch run we initially planned is not justified by the data.

---

## 6. Open question motivating next experiment

**Why didn't the model learn density?** Most likely explanation: training used `Option A` from our earlier planning — duet MIDIs tokenized as a combined stream, no explicit separation between primo and secondo. The model learned the joint distribution of "what notes appear in a Brahms duet" but did not learn "given primo at time t, here is the density of secondo events." That information is what AMT's anticipation framework was designed to capture — and we didn't expose the LoRA to it during training.

---

## 7. Next experiment — Option B / Step-7

**Pre-process duet MIDIs into separate primo and secondo streams, then train LoRA with anticipation framing:**
- Primo tokens prefixed with `CONTROL_OFFSET` (treated as anticipated melody controls)
- Secondo tokens as the prediction target
- Training format matches inference format exactly

**Hypothesis:** model learns to generate secondo *conditioned on* the density of primo, which should resolve the density gap.

**Estimated effort:** several hours of code + ~30-minute training run.

---

## 8. Side notes worth recording

- **Adapter size on disk:** ~2 MB per LoRA adapter (very portable)
- **Training time:** ~5 min for 50 epochs / 850 steps; ~50 min for 500 epochs (with Drive checkpointing overhead)
- **Peak VRAM during training:** 15.6 GB
- **Inference time:** ~6 seconds for one 20s generation
- **Soundfont:** switched from Colab's default FluidR3_GM.sf2 to MuseScore_General.sf2 for more realistic piano timbre
