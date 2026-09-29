# ShouldYouEvenNN
_Stop wasting epochs. Forecast first._
ShouldYouEvenNN is a lightweight, forecasting-driven neural architecture evaluation framework designed to answer one question:
**Is it even worth training a neural network for this task — or will a simpler model do better?**

Instead of fully training deep models and hoping for the best, this project uses early learning signals and curve extrapolation to predict whether a neural architecture is likely to beat classical models like XGBoost or SVM. If not, it gets discarded — fast.

## Core Idea
_Train just enough to know whether to keep going._
Use minimal batches to collect early validation metrics

Fit a learning curve forecaster (e.g., polynomial regression)

Compare predicted final performance against non-neural baselines

Discard unpromising models early — before wasting more compute

This is integrated into a simplified NAS loop (EBE-NAS), built for low-resource environments where smart filtering matters more than exhaustive search.

## Features
🔁 Epoch-by-epoch evaluation under strict batch/instance budgets

🧮 Forecasting module for extrapolating neural performance

⚖️ Baseline-aware discarding, with classical ML models as benchmarks

🧬 Simple NAS engine for evolving MLP-like architectures

📉 Result tracking and diagnostic tools for forecast vs. real score

## Structure Overview

ShouldYouEvenNN/
├── src/                   # All core logic: training, forecasting, discarding, etc.
├── notebooks/             # Visualizations and analysis
├── experiments/           # Datasets, config files, result logs
├── docs/                  # Diagrams, architecture, write-ups
├── main.py                # Entry point to run the EBE-NAS loop
├── config.yaml            # Customizable settings
└── README.md              # This file

## Use Cases
Decide early whether deep learning is necessary

Reduce compute waste in AutoML/NAS workflows

Run architecture search on limited hardware

Teach model selection and resource-awareness
