# AdaBoost from Scratch — Step-by-Step Demo

## Overview

This project implements the **AdaBoost (Adaptive Boosting)** algorithm from scratch using decision stumps (single-level decision trees). It walks through each iteration of the algorithm manually, making it ideal for learning how AdaBoost works under the hood.

AdaBoost is an **ensemble learning** method that combines multiple weak classifiers into a single strong classifier. Each weak learner focuses on the mistakes of the previous one by adjusting sample weights.

## How the Algorithm Works

### Step 1 — Initialize Sample Weights
Every data point starts with an equal weight:

$$w_i = \frac{1}{N}$$

where $N$ is the number of samples (10 in this demo).

### Step 2 — Train a Weak Learner (Decision Stump)
A `DecisionTreeClassifier(max_depth=1)` is trained on the current weighted dataset. A stump makes a single split — it's intentionally a weak classifier.

### Step 3 — Calculate Model Weight (Alpha)
The weight of each weak learner is based on its error rate:

$$\alpha = \frac{1}{2} \ln\left(\frac{1 - \epsilon}{\epsilon}\right)$$

- Low error → high $\alpha$ (model gets more say)
- High error → low $\alpha$ (model gets less say)

### Step 4 — Update Sample Weights
Misclassified samples get **higher** weights, correctly classified get **lower** weights:

- Correct: $w_i \leftarrow w_i \cdot e^{-\alpha}$
- Incorrect: $w_i \leftarrow w_i \cdot e^{\alpha}$

Weights are then normalized to sum to 1.

### Step 5 — Resample the Dataset
A new training set is created by sampling from the original data using the updated weight distribution (cumulative sum method). This forces the next stump to focus on previously misclassified points.

### Steps 2–5 are repeated for each boosting round (3 rounds in this demo).

### Final Prediction
All stumps vote, weighted by their $\alpha$ values:

$$H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t \cdot h_t(x)\right)$$

**Example from the demo:**
| Query   | Stump 1 | Stump 2 | Stump 3 | Weighted Score | Final Prediction |
|---------|---------|---------|---------|----------------|------------------|
| [1, 5]  | 1       | 1       | 1       | +1.10          | **+1 (Class 1)** |
| [9, 9]  | 0       | 0       | 0       | -0.25          | **-1 (Class 0)** |

## Use Cases

AdaBoost is well-suited for:

| Use Case | Description |
|----------|-------------|
| **Binary Classification** | Spam detection, fraud detection, medical diagnosis (disease vs. no disease) |
| **Face Detection** | The Viola-Jones face detection framework is built on AdaBoost with Haar features |
| **Customer Churn Prediction** | Identifying customers likely to leave a subscription service |
| **Credit Scoring** | Classifying loan applicants as low-risk or high-risk |
| **Sentiment Analysis** | Positive vs. negative text classification |
| **Anomaly Detection** | Identifying outliers in network traffic or financial transactions |
| **Bioinformatics** | Gene expression classification, protein function prediction |

### When to Use AdaBoost
- You have a **binary classification** problem
- Individual features are weak predictors on their own
- You want an interpretable ensemble (each stump is a simple rule)
- The dataset is **not too noisy** (AdaBoost is sensitive to outliers)

### When NOT to Use AdaBoost
- **Highly noisy data** — AdaBoost upweights misclassified points, so outliers get amplified
- **Very large datasets** — sequential training can be slower than parallelizable methods like Random Forest
- **Multi-class with many classes** — works best for binary; multi-class extensions exist but add complexity

## Dataset

A simple 2D synthetic dataset with 10 points and 2 features (`X1`, `X2`) with binary labels (0/1).

## Project Structure

```
adaboost_demodata.py   # Main script — full AdaBoost walkthrough
requirements.txt       # Python dependencies
.gitignore             # Files excluded from version control
README.md              # This file
```

## Setup & Run

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python adaboost_demodata.py
```

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- mlxtend
