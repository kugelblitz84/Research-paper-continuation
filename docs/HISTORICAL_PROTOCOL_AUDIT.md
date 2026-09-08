# Historical protocol audit

Recorded before experimental implementation, 2026-09-08. Source paths below are ZIP members under research_paper/. Documents are historical summaries/future proposals, not authorization to extend scope. The user's task is baseline reconstruction and handover before real training.

Selected reference: Phase06 Flat B0 CE, Phase11 Flat DenseNet121 CE, Phase02 other Flat backbones; Phase03 Shared three-task training with Phase06 architecture adapter. The older independent cascade and Flat focal candidate are excluded.

| Parameter | Historical value | Source | New location | Verification / notes |
|---|---|---|---|---|
| Backbones | efficientnet_b0,densenet121,densenet169,resnet50,mobilenet_v3_large,efficientnet_b2,efficientnet_b3 | src/models/classification_backbone.py | src/models/backbones.py | Source verified |
| Split | Exact isic2019_train_val_test_split_seed42.csv | data/manifests | data/manifests | Preserve bytes, never regenerate |
| Seed | 42; group split 70/15/15 | frozen manifest | src/reproducibility.py | Same seed does not recreate split |
| Inclusion | split_included=1 AND include_stage_1=1 | src/data/isic2019_dataset.py | src/data.py | Actinic keratosis excluded; frozen exclusions retained |
| Endpoint order | non_malignant,melanoma,bcc,scc | same | configs/protocol.yaml | NM includes nevus, benign keratosis, dermatofibroma, vascular lesion |
| Cohort counts | train17124, validation3668, test3668 | shared config/final reports | src/data.py | Verify from bytes |
| Shared pool | ISIC train + 594 T-category images; natural concat shuffle | src/data/shared_three_task.py | src/data.py | Actual reference has three tasks |
| Shared masks | NM=[1,0,0]; malignant=[1,1,0]; T-category=[0,0,1]; missing=-100 | same | src/data.py; src/losses.py | Mask before loss |
| Shared architecture | One encoder; Dropout(.2)+Linear to 2/3/5 | src/models/shared_three_task.py | src/models/systems.py | Task3 affects training/selection, not endpoint routing |
| Flat architecture | Dropout(.2)+Linear4; MobileNet retains Linear960->1280/Hardswish projection | src/models/phase02_backbones.py and baseline builders | src/models/systems.py | Preserve historical exception |
| Initialization | torchvision weights DEFAULT; all parameters trainable | model builders | src/models/backbones.py | Record actual enum/version |
| Pooling | DenseNet final ReLU + average pool; others native pool | same | src/models/backbones.py | Feature dims1280/1024/1664/2048/960/1408/1536 |
| Train transforms | v2 ToImage; RRC224 scale(.85,1) ratio(.9,1.1) bilinear antialias; H/Vflip.5; rotation15 bilinear; jitter(.1,.1,.1,.02); float32/255; normalize | src/data/transforms.py | configs/protocol.yaml; src/transforms.py | Preserve order |
| Eval transforms | Resize shorter side256 bilinear antialias; center crop224; float32/255; normalize | same | src/transforms.py | All backbones224 including B2/B3 |
| Normalize | mean .485/.456/.406; std .229/.224/.225 | same | configs/protocol.yaml | ImageNet |
| Flat loss | Unweighted CE, no weighted sampler | Phase06 Flat CE config | src/losses.py | Focal not selected |
| Shared losses | Task1 CE; Task2 CB focal beta.9999 gamma2; Task3 weighted CE | Phase03 shared config | configs/protocol.yaml; src/losses.py | Equal1:1:1 task weights |
| Task2 weights | .3485376280807543,.4553489231324597,2.196113448786786 | same | configs/protocol.yaml | Training counts3164/2327/440 |
| Task3 weights | .063475735584673,.12246677245955931,.682845034319967,2.253388613255891,1.8778238443799093 | same | configs/protocol.yaml | Counts355/184/33/10/12 |
| Aggregation | Sum active mean losses / active weight sum | src/training/shared_three_task.py | src/losses.py | Inactive tasks omitted from denominator |
| Optimizer | AdamW lr.0003 wd.0001 default betas/eps | both training implementations | src/training.py | No clipping/freeze |
| Scheduler | CosineAnnealingLR T_max30 eta_min.000001; after validation | same | src/training.py | Last state after scheduler |
| Budget | 30epochs patience7; strict greater improvement | same | src/training.py | Ties consume patience |
| Loader | batch64 workers4 pin/persistent true prefetch2 drop_last false; seed42 per loader | data/dataloaders.py; shared_three_task.py | src/data.py | Explicit workers0 only for smoke |
| AMP | CUDA fp16 train; Flat validation AMP; Shared validation fp32 | training/engine.py; shared_three_task.py | src/training.py | Preserve asymmetry |
| Selection | Flat endpoint val macroF1; Shared mean task1/2/3 val macroF1 | historical trainers | src/training.py | Never endpoint test |
| Hard gate | Stage1 argmax0->endpoint0, else Stage2 argmax+1 | evaluation/phase04_comparative_harness.py | src/routing.py | Four-class endpoint |
| Oracle | True NM->0; true malignant->Stage2 argmax+1 | same | src/routing.py | Diagnostic only |
| Metrics | Fixed labels zero_division0; accuracy/balanced accuracy/macro/weightedF1/perclass/confusion | evaluation/classification_metrics.py | src/metrics.py | Prediction artifacts required |
| Bootstrap | 10000 paired truth-stratified; seed42; percentile95%; numpy linear quantiles | analysis/phase04_final_internal_test.py; stored_prediction_statistics.py | src/statistics.py | Direction hierarchy-minus-flat |
| McNemar | Exact two-sided binomial discordant correctness | same | src/statistics.py | Accuracy secondary |
| Determinism | Python/NumPy/Torch seeds; algorithms warn_only; cuDNN deterministic/benchmark false | utils/reproducibility.py | src/reproducibility.py | No cross-environment bitwise guarantee |

Historical B0 shared reference: epoch6, selection score0.6326175865948461, task scores .7624178951977814/.6987503899226807/.4366844746640764; evidence reports/phase06_multi_backbone_shared/gate06a_existing_evidence_audit.md. Shared checkpoint hash2f1c2393c5c9de15dfa4a1a132a31b9a5b8ede07d7ed6e07ab90918fc2aaa9eb. No checkpoints are included in ZIP.

New resumable checkpoint generations, registry, config snapshots and notebooks are engineering changes. New runtime never imports historical code. Shared has an auxiliary third head to preserve training, not a newly implemented integrated T-category endpoint.
