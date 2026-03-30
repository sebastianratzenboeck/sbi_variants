# Age Imbalance Notes

This note summarizes what has been tried so far to address the strong class imbalance in stellar age, especially the scarcity of young stars.

## Problem

The training distribution is dominated by older stars. In the baseline single-model setting, this led to poor performance on young stars, especially for `logAge` and `m_init`, even when global calibration looked acceptable.

## What We Already Tried

### 1. Joint curriculum sampling

In [train_sbi_posterior.py](/n/home12/sratzenboeck/code/sbi_variants/train_sbi_posterior.py), training can use a joint `(logAge, m_init)` curriculum sampler instead of natural-frequency sampling.

This helps because:
- rare age-mass regions are sampled more often
- young stars are revisited more frequently than under the raw data distribution

### 2. Fixed-size curriculum epochs

We started runs with:
- `--curriculum-epoch-size 5000000`

This avoids defining an epoch as one pass over the full training set, which would otherwise be dominated by old stars.

### 3. Uniform-over-bin style curriculum

We used settings such as:
- `--joint-curriculum`
- `--tau-max 0.0`

This pushes sampling closer to uniform over active bins, which increases exposure to rare young-star regimes.

## What We Added For Evaluation

### 4. Young-star-specific evaluation

We explicitly evaluated performance on stars with:
- `logAge < 7.8`

This made it clear that the single global model fails badly in the young regime even when aggregate metrics look reasonable.

### 5. Balanced age evaluation subset

Using [make_age_regime_splits.py](/n/home12/sratzenboeck/code/sbi_variants/make_age_regime_splits.py), we created a fixed evaluation subset with:
- all available young test stars
- `100,000` mid-age stars
- `100,000` old stars

This prevents old stars from dominating reported evaluation metrics.

## What We Added In The New Age-Mixture Path

### 6. Frozen age-regime train/val splits

We created fixed regime-specific train/val splits for:
- `young`
- `mid`
- `old`

These are used consistently for age-gate and expert-model experiments.

### 7. Balanced age-gate training

In [train_age_gate.py](/n/home12/sratzenboeck/code/sbi_variants/train_age_gate.py), the gate is trained with a custom balanced class sampler.

This means each epoch gives comparable exposure to:
- young
- mid
- old

rather than letting the gate collapse toward the dominant old-star prior.

### 8. Soft gate instead of hard routing

The age gate is treated as a soft classifier, not a hard router.

This matters because if the observations carry weak age information:
- the gate can stay uncertain
- multiple age experts can contribute
- the final posterior can remain broad instead of collapsing onto one age regime

### 9. Separate expert models by age regime

We set up separate NF experts for:
- `young`
- `mid`
- `old`

This avoids forcing a single global posterior model to absorb all age regimes under extreme imbalance.

## Current View

We have addressed age imbalance in three complementary ways:
- curriculum-based oversampling during posterior training
- balanced and regime-specific evaluation
- explicit age-regime factorization with a soft gate and regime experts

## Things Not Yet Tried

Potential next steps if the current setup is still insufficient:
- class-weighted posterior losses
- focal loss for the age gate
- extra oversampling within the young expert
- a binary `young vs not-young` gate before the 3-way gate
