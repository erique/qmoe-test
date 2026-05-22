# qmoe-test

Cross-architecture, cross-bit-width study of QMoE-style ternary compression on
MoE language models. Three models, four bit widths (fp16 / 1.5-bit ternary /
2-bit / 3-bit), end-to-end perplexity on a fixed held-out C4 sample.

## Headline result

PPL on 65,504 held-out C4 tokens (32 samples × 2048):

| model            | params | fp16 | 1.5-bit | 2-bit | 3-bit |
|------------------|-------:|-----:|--------:|------:|------:|
| Mixtral-8x7B-base|  47 B  | 6.61 |    1165 |   687 |  7.44 |
| DSV2-Lite-base   |  16 B  | 8.80 |    25.0 |  14.1 |  9.27 |
| DSV3-Base        | 671 B  | 7.42 |   13.77 |     — |  7.60 |

Two takeaways:

1. **Architecture matters dramatically** — DeepSeek's many-small-experts +
   shared-expert + top-k≥6 routing compresses ~80× better than Mixtral at
   the same bit width.
2. **Scale matters too, but not enough alone to clear ship-quality at the
   QMoE codec's design point (1.5-bit ternary).** V3 1.5-bit lands at PPL
   13.77 — functional, observably degraded. 3-bit GPTQ on V3 lands at PPL
   7.60 — ΔPPL +0.18 vs fp16, drop-in ship quality.

## What's in this repo

- `PROGRESS.md` — full chronological log: phase-by-phase results, optimisation
  notes, prediction-vs-measurement reconciliation, and a `Setup` section with
  the exact commands to reproduce the numbers.
- `.gitignore` — excludes bulk artefacts (model weights, calibration outputs,
  run logs, the V3 NVMe symlink).

The harness code (Mixtral and DeepSeek-V2/V3 PPL harnesses, ShardLoader fp8
dequant, lazy expert cache, batched-across-experts GPTQ) lives in `qmoe/`
as a git submodule of this repo, pointing at the
[`erique/qmoe`](https://github.com/erique/qmoe) fork on branch `moe-ppl-eval`.
Clone with `--recurse-submodules` to get everything in one go:

```sh
git clone --recurse-submodules https://github.com/erique/qmoe-test.git
```

## Status

This is a study, not a library. The deliverable is the report (`PROGRESS.md`)
plus the harness on the qmoe fork. No runtime integration, no shippable
compressed-model artefact — that would be a separate project. See
`PROGRESS.md` § *Setup* for reproduction instructions.
