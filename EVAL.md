# Evaluation: human vs answering_machine on real call audio

Prompt-level tuning of the vLLM-Omni classifier (Qwen3-Omni Thinker at
`5.129.212.83:51005`, served as `omni-model`) for the binary task
**human vs answering_machine**, measured on a held-out split. The model was not
changed — only the prompt and `max_tokens`.

## Task

Outbound sales calls where **the operator is our automated voice agent** (it
leads, introduces itself, asks questions). We classify the **client** (the party
that answered): a live person or an answering machine / voicemail. Gold labels
come from the dataset folders (`dataset/human/`, `dataset/ao/`); labels are
authoritative. Output is binary — `unknown` is not used.

Caveat that shaped the result: operator and client are **mixed into a single
mono stream** (all 147 files are mono duplicated into fake stereo, L≡R), so the
model must reason about both speakers jointly — the client cannot be isolated.

## Data & split

147 samples: 84 human, 63 answering_machine. Deterministic **stratified
train/dev/test = 60/20/20** by a hash of the sample id (decorrelated from
filename/source order):

| split | human | answering_machine |
|-------|------:|------------------:|
| train | 50 | 37 |
| dev   | 17 | 13 |
| test  | 17 | 13 |

Discipline: prompts were iterated looking **only at dev**; **test was scored
once** at the end. `max_tokens` was tuned on dev, so the test number is the
honest estimate (dev is mildly optimistic).

## Harness (`tools/`)

- `build_manifest.py` — scan `dataset/` → manifest CSV with gold + split.
- `make_config.py` — generate a run config for a given system prompt / few-shot.
- `score.py` — confusion matrix, per-class P/R/F1, accuracy, macro-F1; `--split`, `--list-errors`.
- `characterize.py` — diagnostic pass (subtype + acoustic cues per sample).

Reproduce:

```bash
WS=/tmp/eval
python tools/build_manifest.py --root dataset --out $WS/manifest.csv
python tools/make_config.py --system prompts/system_v2_acoustic.txt \
  --manifest $WS/manifest.csv --out-jsonl $WS/v2.jsonl --out $WS/v2.yaml --max-tokens 2048
python audio_classifier_service.py --config $WS/v2.yaml
python tools/score.py --pred $WS/v2.jsonl --manifest $WS/manifest.csv --split test --list-errors
```

## Results

Same model throughout; only the prompt / token budget changed.

| variant | split | accuracy | macro-F1 | AO recall | human recall |
|---------|-------|---------:|---------:|----------:|-------------:|
| baseline (original prompt) | dev  | 0.633 | 0.529 | 0.154 | 1.000 |
| baseline | test | 0.567 | 0.482 | 0.077 | 0.941 |
| **v2 (acoustic, max_tokens 2048)** | dev  | **0.933** | **0.931** | 0.846 | 1.000 |
| **v2 — FINAL (held-out)** | **test** | **0.867** | **0.874** | **0.769** | 0.941 |

**Winning prompt: `prompts/system_v2_acoustic.txt`.** Core idea: *any automated
or recorded response is `answering_machine`, even if the voice sounds human*;
weigh acoustic + interactivity cues, not just the words. Overall AO recall rose
from ~8% to 49/63 ≈ 78%.

Two fixes mattered:
1. **Prompt reframe** (acoustic-first machine definition) — the bulk of the gain.
2. **`max_tokens` 512 → 2048** — the Thinker's `<think>` block was truncated
   before the answer on hard cases (14 `invalid_answer` at 512 → 0–1 at 2048).

## What was tried and did NOT work (documented dead ends)

- **Operator/client role-framing in the prompt** ("the operator is our bot,
  judge only the client"): **collapsed AO recall to 0.0** on dev (macro-F1 0.36),
  twice. Once told one speaker is a bot, the model attributes any human-sounding
  voice to "the client" and defaults to human. The model cannot do reliable
  speaker-role reasoning from mixed mono via prompt alone.
- **Audio few-shot examples**: the endpoint returns **HTTP 503** on multi-audio
  requests — not viable here.
- **Channel separation** (feed only the client's channel): **impossible** — all
  147 files are mono duplicated into fake stereo (L≡R, corr = 1.0), so operator
  and client cannot be separated. Note: the stereo *mode* metadata spuriously
  correlates with class (human=stereo, ao=joint_stereo) — an encoder artifact,
  not a usable signal.

## Residual errors — the ceiling

14 of 63 AO are still missed (see **`ao_missed_by_v2.csv`**). The dominant
failure is high-quality outbound **voice-AI clients**: the diagnostic pass rates
several of them `human` with confidence 0.95 (natural prosody, turn-taking,
interruption handling). When the model's own oracle can't tell them from a live
human, prompt tweaks can't reliably fix them — this is the model's perceptual
ceiling on this data, not a prompt bug. A smaller subset (studio-clean +
scripted, `machine_mimic`) is catchable and is where any further prompt work
should focus.
