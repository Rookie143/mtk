# Open-Source Release Checklist

Use this checklist before publishing `adaptive_attack`.

## Must decide before release

- [ ] Add a project license file, for example `LICENSE` with MIT, Apache-2.0, or the license selected by the project owners.
- [ ] Add citation metadata, either `CITATION.cff` or a BibTeX entry in `README.md`.
- [ ] Decide where to host large artifacts:
  - model weights are not committed;
  - `reference_bank.pt` is not committed;
  - raw reproduction outputs are not committed.
- [ ] Publish checksums for released `.pt` artifacts.
- [ ] State the exact sample count used for each reported table.

## Reproduction transparency

- [ ] Report raw counts with percentages, e.g. `ASR = 18/30 = 60.0%`.
- [ ] Keep `raw_results.jsonl` privately archived for auditability.
- [ ] Public logs should be reviewed before release because `generation` may contain unsafe model outputs.
- [ ] If using a detector subset, report `--detector-max-anchors`; otherwise state that the full reference bank was used.
- [ ] If claiming paper-level reproduction, use the paper-level attack sample count, detector calibration protocol, and reference-bank composition.

## Local files that should not be committed

- `*.pt`
- `*.jsonl`
- `repro_*/`
- `__pycache__/`
- model checkpoints

The included `.gitignore` excludes these by default.
