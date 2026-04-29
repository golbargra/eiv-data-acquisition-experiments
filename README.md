# EIV Data Acquisition Experiments

This repository contains experiments for my Errors-in-Variables data acquisition research.

## Goal

The experiment compares three settings:

1. Clean benchmark
2. Naive noisy model
3. Two-stage denoised model

We vary:

- noise level
- labeled sample size n
- unlabeled sample size m

## Dataset

The dataset can be downloaded from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading, place `creditcard.csv` in a `data/` folder.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
