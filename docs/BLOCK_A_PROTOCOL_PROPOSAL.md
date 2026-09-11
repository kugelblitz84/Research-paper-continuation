# Proposed H2-H4 protocol extension - awaiting researcher decision

The current `configs/protocol.yaml` and its lock define:

- H1: three task heads, including T-category, with masked task losses.
- H1 selection: mean Task-1/Task-2/Task-3 validation macro-F1.
- Flat selection: four-class endpoint validation macro-F1.
- No H2, H3, or H4 checkpoint-selection rule.

The user's requested H2-H4 designs contain only Task 1 and Task 2. Neither dropping
H1's third task nor silently treating its selection mean as endpoint performance
would preserve the scientific baseline.

Proposed explicit extension for new systems only:

1. Use the requested two-task architectures. Keep H1 unchanged.
2. Retain the locked Task-1 cross-entropy and Task-2 class-balanced focal formula,
   weights, gamma and active-task normalization. No T-category loss for H2-H4.
3. Use the frozen ISIC train/validation memberships and all existing preprocessing,
   augmentation, initialization, head, optimizer, schedule, epoch and seed settings.
4. Select H2/H3/H4 checkpoints by four-class endpoint validation macro-F1. Soft
   systems use the unconditional product distribution; H3 uses hard routing.
5. For dedicated systems, train independent networks within the existing joint
   epoch loop and select one paired epoch using the endpoint score. Export each
   component's immutable checkpoint and hash plus the complete joint resume state.
   Task-2 training forwards only malignant-labelled samples.
6. Record these new-system rules in a separately versioned, digest-checked extension
   included in each new resolved config. Keep both existing baseline config hashes
   and the original protocol lock unchanged.
7. Disclose H1's retained third task and its original selection rule in later
   scientific comparisons. This infrastructure does not claim to remove that
   difference or establish empirical superiority.

No extension has been activated. The generic plumbing is tested with explicitly
synthetic configurations. Real H2-H4 execution raises a clear error before data
access until the researcher chooses this extension or specifies a different policy.
