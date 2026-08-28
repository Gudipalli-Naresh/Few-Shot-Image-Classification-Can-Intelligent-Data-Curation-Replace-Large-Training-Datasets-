# Reproducibility

How to re-run this project and obtain the results reported in the README and the dissertation.

---

## 1. Environment

Developed and run entirely on **Kaggle notebooks with a T4 GPU accelerator**.

```
Python 3.11
torch, torchvision      (CUDA build; CPU works but is substantially slower)
numpy
scikit-learn            KMeans, NearestNeighbors, F1 / confusion metrics
scipy                   spearmanr, ttest_rel, pearsonr
matplotlib
```

```bash
pip install torch torchvision numpy scikit-learn scipy matplotlib
```

No other dependencies. No code from third parties is included in this repository the
forgetting-event and coreset methods were implemented from the descriptions in their respective
papers (see the Key Papers table in the README), not copied from the authors' repositories.

---

## 2. Datasets

Both datasets are standard public benchmarks and download automatically on first run no
manual step is required.

| Dataset | Source | Obtained by |
|---|---|---|
| MNIST | LeCun et al. | `torchvision.datasets.MNIST(root=..., download=True)` |
| CIFAR-10 | Krizhevsky | `torchvision.datasets.CIFAR10(root=..., download=True)` |

Raw dataset files are **not** committed to this repository (`.gitignore` excludes them). On
Kaggle, CIFAR-10 downloads slowly; staging it by symlink from an existing notebook output is
faster than re-downloading, and torchvision MD5-verifies the archive either way, so the worst
case is a slow download rather than corrupted data.

---

## 3. Run order

Later phases consume earlier phases' outputs, so order matters.

| Step | Phase | Depends on | Produces |
|---|---|---|---|
| 1 | `phase1-baseline` | Raw datasets, random seeds, training hyperparameters | Five per-seed checkpoints (one per seed, per dataset) + the ceiling results |
| 2 | `phase2-random-subsampling` | Nothing from Phase 1 | The random-selection floor |
| 3 | `phase3-class-balanced` | Nothing from Phase 1 | Class-balanced results, and the per-class quota used by Phases 4–5 |
| 4 | `phase4-k-means` | Phase 1 checkpoints (for embeddings) | K-means coreset results, selection indices |
| 5 | `phase-5-hard-examples-mining` | Phase 1 checkpoints (for forgetting scores) | Hardest-first results, plus the Phase 5b difficulty-axis arms |

**Phase 1 must run first.** Its checkpoints are read-only inputs to Phases 4 and 5; those phases
will not run without them.

---

## 4. Checkpoints

Phase 1 writes five checkpoints per dataset, named:

```
baseline_mnist_seed{42,123,456,789,1011}_model.pth
baseline_cifar10_seed{42,123,456,789,1011}_model.pth
```

Note `cifar10`, not `cifar`. Later phases locate these with a recursive `glob()` search rather
than a hardcoded path, so they can be attached from any Kaggle input directory without editing
code. If running locally, place them anywhere under the search root and the glob will find them.

Checkpoints are **not committed to this repository** (`.gitignore` excludes `*.pth`) — they are
binaries, and regenerating them is a documented, deterministic step. To recreate them, run
Phase 1's training path in full.

---

## 5. Determinism and what actually reproduces

Every source of randomness is seeded:

```python
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# plus seed_worker() for DataLoader worker processes, which are separate
# processes with independent RNG state
```

**What reproduces exactly:** all CIFAR-10 results, across every phase. Re-running the Phase 5
`hard` arm reproduces its locked values (10.11 / 9.89 / 15.01 / 22.42 / 38.06) byte-identically.

**What does not:** MNIST results for the difficulty-based phases (5 and 5b). Three independent
runs of identical code produced three different result sets — the largest discrepancy being
17.92 pp at the 0.5% subset. The cause is structural rather than a coding error: MNIST converges
so completely that almost all forgetting counts collapse to 0 or 1, so the difficulty ranking is
dominated by random tie-breaking rather than genuine signal. Small platform-level
non-determinism (already documented in Phase 1, where two runs of the ceiling gave 98.98% vs
99.16%) is amplified by this into large swings in which examples fall into the thin never-learned
tier.

**Consequence:** all quantitative claims in this project rest on CIFAR-10. MNIST results for
Phases 5 and 5b are reported for completeness and should be read as indicative, not as exact
reproducible values.

---

## 6. Configuration integrity

Every phase serialises its fixed constants, hashes them, and embeds the first ten hex characters
of that hash in the filename of any cached statistics it writes:

```python
CONFIG_HASH = hashlib.md5(
    json.dumps(CONFIG_FIELDS, sort_keys=True).encode()).hexdigest()[:10]
```

`sort_keys=True` makes the hash independent of dictionary insertion order. If any constant were
edited, the hash would change, every cache lookup would miss, and the change would surface
immediately as a re-run rather than silently as a contaminated comparison.

**The hash `db554d58b3` has held unchanged from Phase 1 through Phase 5b.** This is the strongest
single piece of evidence that no experimental constant drifted across the project, which matters
because every finding here is a comparison *between* phases.

---

## 7. Fixed constants

Identical in every phase. Changing any of these invalidates cross-phase comparison.

| | |
|---|---|
| Architecture | SimpleCNN 3 conv blocks (32→64→128), AdaptiveAvgPool2d(2,2), FC 512→256→10 |
| Parameters | 227,018 (MNIST, 1 channel) · 227,594 (CIFAR-10, 3 channels) |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau(mode='min', factor=0.5, patience=3), steps on **training** loss |
| Loss | CrossEntropyLoss (no explicit softmax in the model the loss applies it) |
| Dropout | 0.5 |
| Seeds | [42, 123, 456, 789, 1011] |
| Subset sizes | 0.2%, 0.5%, 1%, 2%, 5% |
| Epochs | MNIST 20 · CIFAR-10 30 |
| Batch size | 128 (`min(BATCH_SIZE, len(subset))` for very small subsets) |
| Workers | 2, with `seed_worker` |
| MNIST normalisation | mean 0.1307, std 0.3081 no augmentation |
| CIFAR-10 normalisation | mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010) |
| CIFAR-10 augmentation | RandomHorizontalFlip + RandomCrop(32, padding=4) **train split only** |


---

# AI Prompts

This file records the prompts used with Anthropic Claude during development, one per
experimental phase. They are reproduced as sent, unedited, so the record reflects what was
actually asked rather than a tidied-up version.

These are working prompts rather than a formal protocol. Each phase's prompt carries forward the
constants and the *results* of the phases before it, because that is how the project was run:
one phase at a time, with the next phase's design informed by what the previous one produced.
Where a prompt states a hypothesis or a verdict rule, that was written before that particular
phase was run, not reconstructed afterwards.

Two prompts contain statements that were correct when written but were later superseded; these
are flagged inline and listed again at the end rather than edited out.

---

## Phase 1: Baseline ceiling

> Walk me through the CNN. why this architecture? more about this and give good narration to
> explain in simple terms

Phase 1 was developed conversationally rather than from a single specification, so this is the
only substantive prompt from that phase.

---

## Phase 2: Random subsampling (the floor)

```
>CURRENT TASK: Phase 2 Random Subsampling Build the experimental harness. For each subset size, randomly sample that fraction of the training data, train SimpleCNN, evaluate across 5 seedsand record mean std. This is the FLOOR the number intelligent curation must beat in Phases 3, 4, 5. The harness built here gets reused by all later phases, so build it clean.

>WHAT PHASE 2 MUST PRODUCE:

>Accuracy vs subset size curves for MNIST and CIFAR-10 (with std error bars across 5 seeds)
Per-class F1 at each subset size
>Mean std table across all 5 seeds, per subset size, per dataset .
A clean reusable sampling + train + evaluate harness that later Phases can be used.
FIXED CONSTANTS --- never change
Architecture: SimpleCNN --- 3 conv blocks (Conv→BatchNorm→ReLU→MaxPool, channels 32→64→128), AdaptiveAvgPool2d(2,2) → Flatten → Linear(512,256) → ReLU → Dropout(0.5) → Linear(256,10). No softmax (CrossEntropyLoss applies it).
Optimiser: Adam lr=1e-3, weight_decay=1e-4
Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=3) --- steps on TRAINING loss. No verbose argument.
Loss: CrossEntropyLoss
Seeds: [42, 123, 456, 789, 1011]
Subset sizes: 0.2%, 0.5%, 1%, 2%, 5%
Batch=128, num_workers=2
Epochs: MNIST=20, CIFAR=30
MNIST norm: (0.1307,)/(0.3081,), no augmentation
CIFAR norm: mean (0.4914,0.4822,0.4465) std (0.2023,0.1994,0.2010), augmentation: RandomHorizontalFlip + RandomCrop(32,padding=4) on train only
worker seeding: seed_worker(worker_id) using torch.initial_seed() % 2**32 --- required for reproducibility with num_workers>0 <17. Platform: Kaggle >GPU T4 x2
```

## Phase 3: Class-balanced selection
```
 FIXED CONSTANTS  never change across any phase
Architecture: SimpleCNN --- 3 conv blocks (Conv→BatchNorm→ReLU→MaxPool, channels 32→64→128),
AdaptiveAvgPool2d(2,2) → Flatten → Linear(512,256) → ReLU → Dropout(0.5) → Linear(256,10).
No softmax (CrossEntropyLoss applies it).
Optimiser: Adam lr=1e-3, weight_decay=1e-4
Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=3)  steps on TRAINING loss. No verbose argument.
Loss: CrossEntropyLoss
Seeds: [42, 123, 456, 789, 1011]
Subset sizes: 0.2%, 0.5%, 1%, 2%, 5%
Batch=128, num_workers=2
Epochs: MNIST=20, CIFAR=30
MNIST norm: (0.1307,)/(0.3081,), no augmentation
CIFAR norm: mean (0.4914,0.4822,0.4465) std (0.2023,0.1994,0.2010)
CIFAR augmentation: RandomHorizontalFlip + RandomCrop(32,padding=4) on train only
Worker seeding: seed_worker(worker_id) using torch.initial_seed() % 2**32
Platform: Kaggle GPU T4 x2
PHASE 1 RESULTS CEILING (never change)
MNIST: Top-1 = 98.98% ± 0.43 | Macro-F1 = 0.9898 ± 0.0042
Per seed: 42→99.14% / 123→98.14% (outlier, retained) / 456→99.21% / 789→99.30% / 1011→99.13%
CIFAR-10: Top-1 = 80.93% ± 0.33 | Macro-F1 = 0.8084 ± 0.0028
Per seed: 42→80.81% / 123→80.58% / 456→81.12% / 789→80.67% / 1011→81.47%
Weakest: cat=0.6465, bird=0.7262, dog=0.7386 | Strongest: automobile=0.9055
PHASE 2 RESULTS FLOOR (locked)
MNIST (ceiling 98.98%):
0.2% (120 imgs): 24.31% ± 10.76% | F1: 0.1672 ± 0.1206
0.5% (300 imgs): 82.03% ± 10.29% | F1: 0.8095 ± 0.1177
1% (600 imgs): 94.54% ± 0.82% | F1: 0.9449 ± 0.0081
2% (1200 imgs): 95.68% ± 1.22% | F1: 0.9565 ± 0.0124
5% (3000 imgs): 96.87% ± 0.46% | F1: 0.9685 ± 0.0046
CIFAR-10 (ceiling 80.93%):
0.2% (100 imgs): 23.29% ± 2.98% | F1: 0.2073 ± 0.0288
0.5% (250 imgs): 33.12% ± 3.06% | F1: 0.3169 ± 0.0364
1% (500 imgs): 40.08% ± 0.25% | F1: 0.3928 ± 0.0086
2% (1000 imgs): 48.00% ± 2.25% | F1: 0.4766 ± 0.0186
5% (2500 imgs): 58.05% ± 1.33% | F1: 0.5772 ± 0.0120
Key finding: CIFAR-10 has a 22.9 pp gap at 5%  this is what Phases 3-5 must close.
>MNIST plateau: from 1% onwards gap to ceiling is <5 pp  architecture limit, not data.
```


## Phase 4: K-means coreset selection

 **Superseded:** this prompt asks for embeddings from "the locked Phase 1 seed-42 checkpoint".
 The implementation instead extracts embeddings per seed, using each seed's own Phase 1
 checkpoint, so that run seed *s* uses the seed*s* checkpoint. The repository README describes
 the implemented behaviour.

```
CURRENT TASK: Phase 4 --- K-means Coreset Selection. WHAT CHANGES IN PHASE 4 (only)

sample_indices(dataset, fraction, seed) -> list[int] --- same fixed signature as Phases 2/3.

Mechanics:

1. Extract 256-d penultimate-layer embeddings using the locked Phase 1 seed-42 checkpoint (the layer before the final Linear(256,10) classifier).

2. Run K-means PER CLASS (not globally) --- this preserves class balance automatically, same as Phase 3, so (Phase 4 − Phase 3) isolates the effect of intelligent selection alone, independent of the balance effect already measured.

3. From each per-class cluster, select the point(s) nearest each centroid --- k = per-class quota (same quota logic as Phase 3: base + remainder distribution).

New pre-step not present in P2/P3: embeddings must be extracted from the Phase 1 checkpoint BEFORE sampling can run. This is a separate function, not part of sample_indices itself.

PRE-COMMIT HYPOTHESIS (stated before seeing results, so it is not post-hoc)

K-means coreset selection should meaningfully close the CIFAR gap in the 1--5% range, because it actively selects representative rather than redundant examples per class --- unlike Phase 3, which controlled for balance but left within-class selection random. If the gap does NOT close, that is itself a real finding: it would suggest coverage/representativeness isn't the missing ingredient, motivating Phase 5's hard-example framing instead.

LOCKED REFERENCE VALUES (do not re-derive, do not re-run source phases)

Phase 1 ceiling: MNIST 98.98% ± 0.43 (F1 0.9898 ± 0.0042) | CIFAR-10
80.93% ± 0.33 (F1 0.8084 ± 0.0028)

Phase 1 CIFAR weakest classes: cat 0.6465, bird 0.7262, dog 0.7386 | strongest: automobile 0.9055

Phase 2 floor (random) --- CIFAR Top-1: 0.2%→23.29% | 0.5%→33.12% | 1%→40.08% | 2%→48.00% | 5%→58.05% (22.9pp gap to ceiling at 5% --- the core scientific opportunity)

Phase 2 floor (random) --- MNIST Top-1: 0.2%→24.31% | 0.5%→82.03% | 1%→94.54% | 2%→95.68% | 5%→96.87%

Phase 3 (class-balanced) --- CIFAR Top-1: 0.2%→23.33% | 0.5%→33.03% | 1%→40.50% | 2%→50.28% | 5%→57.00% (near-zero Δ vs P2 --- balance alone does not close the CIFAR gap)

Phase 3 (class-balanced) --- MNIST Top-1: 0.2%→25.67% | 0.5%→90.42% | 1%→93.98% | 2%→95.10% | 5%→96.12%

Phase 3 finding: class balance is a variance reducer, not an accuracy booster. Cat/bird/dog remain weakest even at 250 balanced images/class on CIFAR --- same ordering as Phase 1 ceiling. This is why Phase 4 exists: the question has sharpened from "how much data" to "which data."

SELECTION-VISUALISATION UTILITY (reuse, do not rebuild)

plot_selection_grid(dataset_name, indices, class_names, title, save_path, n_per_class=5) --- built in Phase 3, transform-free (raw ToTensor only, no crop/flip), validated on real data. Reuse unchanged in Phase 4 --- only indices and title change. Requires get_raw_dataset(dataset_name) helper (also from Phase 3).
```

---

## Phase 5:  Hard-example mining via forgetting events

> **Superseded:** this prompt lists two open items that were resolved after it was written
> the Phase 3 MNIST reconciliation, and the Phase 4 `n_init` check. Both outcomes are recorded
> at the end of this file.

```
CURRENT TASK: Phase 5 --- Hard-Example Mining via Forgetting Events.

Consequence: Phase 5 requires a NEW, fully-instrumented training run --- same 5 seeds x 2 datasets x full epoch schedule (MNIST 20, CIFAR 30) --- purely to LOG per-epoch per-example correctness and compute forgetting counts. This is roughly Phase-1-sized compute, not Phase-4-sized. This is a genuine, deliberate scope difference from Phases 2-4 and should be stated plainly in the write-up, not glossed over.

DECISION ALREADY MADE (do not revisit): stick with Toneva's forgetting-events method exactly as pre-registered, NOT Paul et al. 2021's EL2N score, even though EL2N would be cheaper (computable from a handful of early epochs rather than a full run). Reasoning: the falsifiable prediction below was pre-registered specifically for forgetting events, before Phase 4's results were seen. Swapping the difficulty metric now would break the pre-registration and could look like a post-hoc cost-driven substitution. Compute is not the binding constraint on a T4 (Phase 4 confirmed this). Keep this decision unless Maier explicitly overrules it.

MECHANICS

Fresh instrumented training run per (dataset, seed): train on the FULL dataset (not a subset), logging per-example per-epoch correct/incorrect after every epoch.

Forgetting count per example = number of correct->incorrect transitions across training. Examples never learned correctly at all get a defined sentinel value (Toneva's convention --- verify exact handling in the paper, do not assume).

Rank examples by forgetting count, per class (to preserve the class-balance control from Phase 3 --- same quota logic, k = Phase 3 per-class quota, so Phase 5 - Phase 3 isolates difficulty-based selection alone, same design principle as Phase 4).

Select the HARDEST examples (highest forgetting counts) per class, up to quota.

Train the standard SimpleCNN harness (identical to Phases 2-4) on that selected subset, evaluate, aggregate over 5 seeds.

PRE-REGISTERED FALSIFIABLE PREDICTION (stated before running, do not adjust after seeing results) Hard-example mining should show the OPPOSITE pattern to Phase 4:

WORSE than random/balanced at small subset sizes (0.2-1%) --- hard examples alone are insufficient to learn coarse class structure when data is very scarce.

BETTER at larger subset sizes (5%) --- hard examples become informative once coarse structure is already learnable from the rest. This is grounded in Sorscher et al. 2022 (VERIFIED AT SOURCE, safe to cite): "when data is abundant (scarce), the better pruning strategy is to keep the hard (easy) examples" (direct quote, NeurIPS 2022 paper, confirmed via author-hosted PDF and official poster). NOTE: Sorscher studied general difficulty-based pruning, not forgetting-events specifically --- this is an informed analogy justifying the prediction, not a direct restatement of their result. Phrase it that way in the write-up: "consistent with the regime-dependence Sorscher et al. (2022) found for difficulty-based pruning" --- not "Sorscher et al. showed forgetting-event mining does X."

Verdict rule (pre-committed): CONFIRMED if Phase 5 underperforms Phase 3/4 at 0.2-1% AND outperforms at 5%. Any other pattern (including no effect, like Phase 4) is a valid, reportable, non-confirming result --- not a failure to hide.

LOCKED REFERENCE VALUES --- do not re-derive, do not re-run source phases Phase 1 ceiling: MNIST 98.98% ± 0.43 (F1 0.9898 ± 0.0042) | CIFAR-10 80.93% ± 0.33 (F1 0.8084 ± 0.0028) Phase 1 CIFAR weakest classes: cat 0.6465, bird 0.7262, dog 0.7386 | strongest: automobile
0.9055 Phase 2 floor (random) --- CIFAR Top-1: 0.2%->23.29% |
0.5%->33.12% | 1%->40.08% | 2%->48.00% | 5%->58.05% Phase 2 floor (random) --- MNIST Top-1: 0.2%->24.31% | 0.5%->82.03% | 1%->94.54% | 2%->95.68% | 5%->96.87% Phase 3 (class-balanced) --- CIFAR Top-1:
0.2%->23.33% | 0.5%->33.03% | 1%->40.50% | 2%->50.28% | 5%->57.00% Phase 3 (class-balanced) --- MNIST Top-1: TWO VERSIONS EXIST, UNRECONCILED (see Phase 4 pending item P2): notebook summary:
25.65 / 90.19 / 94.06 / 95.06 / 96.45 JSON used by Phase 4: 25.74 /
90.40 / 94.23 / 95.38 / 96.16 CONFIRM WHICH IS CANONICAL BEFORE PHASE 5 REPORTS ANY MNIST DELTA. Does not affect CIFAR. Phase 4 (K-means coreset) --- CIFAR Top-1: 0.2%->26.16% | 0.5%->34.93% | 1%->42.84% | 2%->49.35% | 5%->56.74% All 10 paired t-tests (P4 vs P3) non-significant. Pre-commit hypothesis "closes the gap" NOT supported. Phase 4 PRIMARY RESULT (the mechanism, pre-registered, confirmed): Spearman(intra/inter ratio, fraction of P1 ceiling recovered at 5%) =
-0.842, p = 0.0022. Tight classes (ship, automobile, truck) benefit from curation; diffuse classes (cat, dog, bird) do not. Phase 4 STATUS: NOT YET FULLY LOCKED. Two blockers outstanding --- confirm resolved before treating as closed: (a) n_init check --- CIFAR 5% re-run with n_init=10 vs n_init=1, to settle whether cluster-init quality mattered. (b) P3 MNIST reconciliation (see above). Phase 5 REUSES: the same 5 seeds. Must NOT reuse Phase 1/4 checkpoints for forgetting counts (those weren't instrumented per-epoch) --- Phase 5's instrumented run is genuinely new compute, separate from Phase 1's original baseline run. Phase 1's own locked ceiling numbers remain the reference; Phase 5 does not touch or recompute them.

SELECTION-VISUALISATION UTILITY (reuse, do not rebuild) plot_selection_grid(dataset_name, indices, class_names, title, save_path, n_per_class) --- built Phase 3, transform-free, validated on real data. Reuse unchanged --- only indices and title change.
```

---
