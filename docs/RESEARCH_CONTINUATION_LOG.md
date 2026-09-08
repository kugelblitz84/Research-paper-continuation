# Research continuation log

## 2026-09-08 ? Baseline reconstruction build

Scientific question: can the historical controlled Flat versus Shared-Hard system
be reproduced through an organized notebook-first implementation without changing
its experimental meaning?

Scope: seven supported backbone pairs, B0-first reproduction gate, no future
Shared-Soft/Dedicated/ensemble/head-ablation systems and no real training yet.

Decisions established from source evidence:

- Preserve the actual three-task shared training pool, third head and three-task
  validation selection. The four-class endpoint only uses tasks 1/2.
- Preserve Flat MobileNet's native projection/Hardswish before Dropout/Linear.
- Preserve the exact frozen manifests; do not regenerate splits from a seed.
- Preserve CE for Flat, task-specific historical shared losses and AMP asymmetry.
- Keep document-reported scores, archived prediction replay and future reconstructed
  results physically and semantically separate.

Build evidence is recorded under `validation/` and summarized in BUILD_HANDOVER.md.
The historical-prediction replay is not a new experiment and adds no training row.
The real experiment registry remains header-only.

Next authorized scope after handover review: Azure image/environment preflight,
then EfficientNet-B0 Flat and Shared-Hard. Record every actual run and evaluation.
The six remaining pairs require explicit acceptance of the B0 reproduction evidence.

## Additive continuation procedure

For every later system: freeze a new config/protocol version; document its question
and changed factor; execute; preserve checkpoint/predictions; evaluate and save
statistics; append registry/continuation records; freeze the result. Add notebook
06 onward. Do not modify the completed baseline to implement a later variant.

No quantitative B0 reproduction tolerance has been supplied. Show actual differences
and uncertainty for researcher review. Do not tune on the consumed internal test.
The future DOCX roadmaps suggest different experiment orders; choosing a future
order remains outside this baseline rebuild.
