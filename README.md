# Welcome

This repository contains reproducible experiments related to "A note
reguarding an extra use of test data."

Of primary concern was the question of how general the score to
confidence method would prove across different datasets, and so the
primary variations investigated were transforms on the dataset.

## Overview

The code operates in three phases: (1) generate a reproducible
variation on the MNIST dataset, (2) train an ensemble of neural
networks, each with relatively poor performance, (3) analyze the
sensitivity of our confidence estimation using the Kolmogorov-Smirnov
test.

### Ensemble members

The individual neural networks have two layers, with 32 hidden units.

> Layer 1: 784 -> 32, relu activation function

> Layer 2: 32 -> 10, softmax activation function

### Poor Neural Network Accuracy

As confidence estimation is of little interest for "solved problems,"
the chosen network structure only achieves approximately 90% accuracy.

### Ensemble training sets

The neural networks are each trained on 90% of the training set data,
with no overlap between held out sets. (ie given an ordering of the
training set, net one is trained with the first 10% of the data held
out, net two is trained with the second 10% held out, etc.)

### Ensemble testing sets

Because the score to confidence function is generated using the test
set, and because this project is primarily concerned with
understanding the properties thereof, we create two test sets.

## How to run

### Setup: Make a new subdirectory

The scripts create cached assets during execution, so start by making
a new directory to hold your experiment's results.

> mkdir my_experiment

> cd my_experiment

### Step 1: Reproducibility

First we generate a static shuffling of the mnist training and test
datasets. The output file ``1_static_shuffle.dat`` will be created. Three
modes are supported:

> julia ../1_reproducibility.jl MIX

With the MIX option, the script combines the training and test data
into a single array of length 70,000, segmenting off two length 10,000
portions for our test1 and test2 datasets.

> julia ../1_reproducibility.jl KEEP

With the KEEP option, the script keeps the mnist test data separate as
the test2 dataset, segmenting one length 10,000 portion for the test1
dataset. This is desirable because the mnist official test set
includes crossed 7s, unlike the training set.

> julia ../1_reproducibility TRIM p cat_1 cat_2 ...

With the TRIM option, the script removes training examples in cat_1,
... cat_n with probability p, operating as KEEP for the test1 and
test2 datasets.

example: julia ../1_reproducibility 0.5 3 4 7

This removes training examples from categories 3, 4, and 7 with a 50%
probability.

### Step 2: Training and running the model

Here we create a 10 neural network ensemble from the training set
created in step 1. Each network is trained independently / without
boosting.

Output files are ``2_model.dat`` which contains a saved copy of the
final networks, and ``2_scored_test_data.dat`` which contains the
saved output of the networks for both test sets in the following
format:

```julia
struct RunResult
  output :: Array{Float32, 1}
  max_output_i :: Int64
  score :: Float32
  correct_category :: Int64
end
```

> julia ../2_train_and_run.jl

### Step 3: Reports on the test1 and test2 performance

> julia ../3_report.jl

Here we compute the Kolmogorov-Smirnov test values, and report the
corresponding maximum alpha values, stored in ``3_ks_stats.csv``.

Additionally, we create ``3_scoring_extra_info.csv`` for use in our
plots, with the following entries for each of the test datasets:

> predicted category, actual category, true_positive, score, gt confidence, lt confidence
