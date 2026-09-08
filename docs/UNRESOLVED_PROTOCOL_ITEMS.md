# Unresolved protocol items

- Pasted two-head outline conflicts with the actual three-task Shared reference. Preserve third auxiliary head, 594 additional train samples, and three-task selection unless user explicitly chooses a new protocol. Clarification was requested; the build follows the source-backed historical three-task interpretation. A two-task variant would require an explicitly new protocol.
- Flat MobileNet retains a 960->1280 Linear/Hardswish projection. Preserve this historical exception.
- No checkpoints or raw image collection in ZIP. Real image/hash checks and reproduction training require dataset setup.
- Final seven-backbone/HIBA synthesis cited by documents is missing. Rounded document scores must remain document-reported, not independently verified.
- ZIP requirements are not a complete historical runtime capture. Record actual local/Azure environments separately.
- No numerical acceptance tolerance supplied. Report historical/new/difference for B0 and review before releasing six remaining pairs. Never tune using internal test.
- Historical persistent worker augmentation RNG is not checkpointed. Resume restores main RNG and loader generators but cannot promise identical uninterrupted multiworker augmentations.
