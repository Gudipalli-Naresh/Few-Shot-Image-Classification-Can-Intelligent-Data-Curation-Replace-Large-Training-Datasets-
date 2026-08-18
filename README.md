# Few-Shot Image Classification: Can Intelligent Data Curation Replace Large Training Datasets?

---
 
## Overview
 
Training deep neural networks on large labelled datasets is expensive in time, money and compute. But not all training examples are equally useful. This project investigates whether intelligently choosing a small subset of training data can approach the performance of training on the full dataset.
 
The experiment is fully controlled: one architecture (SimpleCNN), one optimiser, five fixed random seeds, two datasets (MNIST and CIFAR-10) and five subset sizes from 0.2% to 5%. The only variable across phases is the selection strategy:- Random, class-balanced, diversity-based or importance-based. Because everything else is locked, Any difference in accuracy is attributable to the selection method alone.
 
---
 
## Repository Structure
 
```
├── phase1-baseline/
│   └── phase1-baseline-final.ipynb        # Full-dataset ceiling (COMPLETE ✅)
├── phase2-random-subsampling/
│   └── phase2_random_subsampling.py       # Random subset floor (COMPLETE ✅)
├── phase3-class-balanced/
│   └── phase3-class-balanced.ipynb        # Equal per-class quota (COMPLETE ✅)
├── phase4-kmeans/
│   ├── phase-4-k-means.ipynb              # K-means coreset selection (COMPLETE ✅)
│   ├── results/                           # per-seed results, paired stats, manifest
│   ├── figures/                           # 8 figures
│   └── selected_indices/                  # 50 JSON files — GPU-free reproduction
├── .gitignore
└── README.md
```
 
> Phases 3–5 will be added as each phase completes.
 
---
 
## Experimental Setup (Fixed Across All Phases)
 
| Component | Value |
|-----------|-------|
| Architecture | SimpleCNN 3 conv blocks (32→64→128), AdaptiveAvgPool2d(2,2), FC 512→256→10 |
| Optimiser | Adam lr=1e-3, weight_decay=1e-4 |
| Scheduler | ReduceLROnPlateau (min, factor=0.5, patience=3) steps on training loss |
| Seeds | [42, 123, 456, 789, 1011] 5 seeds for mean ± std |
| Subset sizes | 0.2%, 0.5%, 1%, 2%, 5% of training set |
| Batch size | 128 |
| Epochs | MNIST: 20 · CIFAR-10: 30 |
| MNIST | Normalise only  mean 0.1307, std 0.3081 |
| CIFAR-10 | RandomHorizontalFlip + RandomCrop(32, padding=4) + normalise (train only) |
| Platform | Kaggle GPU T4 x2 |
 
---
 
## Phase Status & Results
 
### Phase 1: Full Dataset Ceiling ✅ COMPLETE
 
Trains SimpleCNN on the complete training set to establish the performance ceiling.
Every later phase is reported as a distance from these numbers.
 
| Dataset | Top-1 Accuracy | Macro-F1 |
|---------|---------------|----------|
| MNIST | **98.98% ± 0.43** | 0.9898 ± 0.0042 |
| CIFAR-10 | **80.93% ± 0.33** | 0.8084 ± 0.0028 |
 
CIFAR-10 weakest classes: cat = 0.6465 · bird = 0.7262 · dog = 0.7386
CIFAR-10 strongest class: automobile = 0.9055
 
---
 
### Phase 2: Random Subsampling Floor ✅ COMPLETE
 
Trains SimpleCNN on randomly sampled tiny subsets. This is the naive baseline the number every intelligent curation strategy in Phases 3–5 must achieve.
 
**MNIST** (ceiling: 98.98%)
 
| Subset | Images | Top-1 Mean | ± Std | Gap to ceiling |
|--------|--------|-----------|-------|----------------|
| 0.2% | 120 | 24.31% | 10.76% | 74.7 pp |
| 0.5% | 300 | 82.03% | 10.29% | 16.9 pp |
| 1% | 600 | 94.54% | 0.82% | 4.4 pp |
| 2% | 1200 | 95.68% | 1.22% | 3.3 pp |
| 5% | 3000 | **96.87%** | 0.46% | 2.1 pp |
 
**CIFAR-10** (ceiling: 80.93%)
 
| Subset | Images | Top-1 Mean | ± Std | Gap to ceiling |
|--------|--------|-----------|-------|----------------|
| 0.2% | 100 | 23.29% | 2.98% | 57.6 pp |
| 0.5% | 250 | 33.12% | 3.06% | 47.8 pp |
| 1% | 500 | 40.08% | 0.25% | 40.9 pp |
| 2% | 1000 | 48.00% | 2.25% | 32.9 pp |
| 5% | 2500 | **58.05%** | 1.33% | 22.9 pp |
 
> CIFAR-10 has a **22.9 pp gap** at 5% between random selection and the full-dataset ceiling. That gap is what Phases 3–5 are designed to close.
 
---
 
### Phase 3: Class-Balanced Selection ✅ COMPLETE
 
Fixes random sampling's most obvious weakness — accidental class imbalance at tiny sizes — by drawing an equal number of examples from every class. Selection *within* each class remains random, so this phase isolates the balance effect alone.
 
**MNIST** (ceiling: 98.98%)
 
| Subset | Images | Top-1 Mean | ± Std | Δ vs Phase 2 |
|--------|--------|-----------|-------|--------------|
| 0.2% | 120 (12/class) | 25.65% | 5.18% | +1.34 |
| 0.5% | 300 (30/class) | 90.19% | 1.72% | +8.16 |
| 1% | 600 (60/class) | 94.06% | 0.92% | −0.48 |
| 2% | 1200 (120/class) | 95.06% | 1.82% | −0.62 |
| 5% | 3000 (300/class) | **96.45%** | 0.81% | −0.42 |
 
**CIFAR-10** (ceiling: 80.93%)
 
| Subset | Images | Top-1 Mean | ± Std | Δ vs Phase 2 |
|--------|--------|-----------|-------|--------------|
| 0.2% | 100 (10/class) | 23.33% | 1.30% | +0.04 |
| 0.5% | 250 (25/class) | 33.03% | 0.96% | −0.09 |
| 1% | 500 (50/class) | 40.50% | 2.75% | +0.42 |
| 2% | 1000 (100/class) | 50.28% | 2.12% | +2.28 |
| 5% | 2500 (250/class) | **57.00%** | 1.44% | −1.05 |

---
 
### Phase 4: K-Means Coreset Selection ✅ COMPLETE
 
The first phase to choose images by their **content** rather than at random. Each training image is passed through the frozen Phase 1 network to obtain a 256-dimensional penultimate-layer embedding. K-means is then run *within each class*, with k set to the Phase 3 per-class quota, and the real image nearest each cluster centroid is kept.
 
Because the per-class quota is identical to Phase 3, the difference (Phase 4 − Phase 3) isolates **intelligent within-class selection** with the balance effect already subtracted out.
 
Method follows the coreset / k-centers framing of [Coleman et al. (2020)](https://arxiv.org/abs/1906.11829). Embeddings are extracted with the evaluation transform (no augmentation) and L2-normalised before clustering; run seed *s* uses the seed-*s* Phase 1 checkpoint.
 
**MNIST** (ceiling: 98.98%)
 
| Subset | Images | Top-1 Mean | ± Std | Δ vs Phase 3 |
|--------|--------|-----------|-------|--------------|
| 0.2% | 120 | 29.29% | 6.69% | +3.64 |
| 0.5% | 300 | 87.30% | 3.43% | −2.89 |
| 1% | 600 | 93.85% | 1.14% | −0.21 |
| 2% | 1200 | 93.03% | 2.97% | −2.03 |
| 5% | 3000 | **97.25%** | 0.65% | +0.80 |
 
**CIFAR-10** (ceiling: 80.93%)
 
| Subset | Images | Top-1 Mean | ± Std | Δ vs Phase 3 | Paired *p* |
|--------|--------|-----------|-------|--------------|-----------|
| 0.2% | 100 | 26.16% | 3.36% | +2.83 | 0.181 |
| 0.5% | 250 | 34.93% | 2.26% | +1.90 | 0.166 |
| 1% | 500 | 42.84% | 1.64% | +2.34 | 0.341 |
| 2% | 1000 | 49.35% | 1.82% | −0.93 | 0.511 |
| 5% | 2500 | **56.74%** | 0.86% | −0.26 | 0.797 |

---
 
### Phase 5: Hard-Example Mining ✅ COMPLETE
 
Scores examples by forgetting events (Toneva et al. 2019)  the number of times an example
transitions from correctly to incorrectly classified during training  and selects the
hardest-scoring examples. Uses all five Phase 1 seed checkpoints (not seed-42 alone). The
diagnostic identified a sentinel-collapse failure mode: at small subset sizes, "hardest"
degenerates into a random draw from examples the network never learned at all, rather than a
meaningful difficulty ranking.

**CIFAR-10** (ceiling: 80.93%)
 
| Subset | Top-1 Mean | ± Std | vs. random (Phase 2) |
|---|---|---|---|
| 0.2% | 10.11% | 0.79% | −13.18 pp |
| 0.5% | 9.89% | 1.45% | −23.23 pp |
| 1% | 15.01% | 1.76% | −25.07 pp |
| 2% | 22.42% | 2.12% | −25.58 pp |
| 5% | 38.06% | 2.35% | −19.99 pp |
 
**MNIST** (ceiling: 98.98%)
 
| Subset | Top-1 Mean | ± Std | vs. random (Phase 2) |
|---|---|---|---|
| 0.2% | 17.09% | 4.83% | −7.22 pp |
| 0.5% | 39.07% | 9.13% | −42.96 pp |
| 1% | 65.43% | 10.21% | −29.11 pp |
| 2% | 70.57% | 10.59% | −25.11 pp |
| 5% | 94.32% | 3.81% | −2.55 pp |
---
 
---
 
## Key Papers
 
| Paper | Summary |
|-------|---------|
| [Toneva et al. (2019)](https://arxiv.org/abs/1812.05159) | Forgetting events — not all training examples are equal |
| [Coleman et al. (2020)](https://arxiv.org/abs/1906.11829) | Selection via Proxy — cheap proxy models can rank example importance |
| [Paul et al. (2021)](https://arxiv.org/abs/2107.07075) | EL2N scoring — importance-based pruning with an over-pruning warning |
 
---
 
## Declared Limitations
 
1. **No validation split**: scheduler watches training loss; consistent across all phases so cross-phase comparisons remain fair, but absolute numbers are affected.
2. **CIFAR augmentation confound**: augmentation inflates effective training data at tiny sizes; kept identical across all phases so comparisons are fair.
3. **AdaptiveAvgPool2d(2,2)**: retains 512 spatial features rather than collapsing to 128; deliberate choice to preserve coarse spatial layout.
---
