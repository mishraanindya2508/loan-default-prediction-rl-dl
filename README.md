# Loan Default Prediction Modeling — TabNet (DL) vs CQL (Offline RL)

Project: Loan default prediction and offline RL decision policy  
Author: Anindya Mishra  
Repository: `loan-default-prediction-rl-dl`  
Purpose: Compare a supervised deep learning approach (TabNet) for default prediction with an offline reinforcement learning approach (CQL) that directly learns a loan approval policy optimized for financial return.



# Quick summary of what is included

- EDA notebook  
  - Data loading, cleaning, feature engineering, class balance checks, and visualizations.
- TabNet notebook  
  - TabNet model training, threshold tuning, evaluation (AUC, F1), and inference examples.
- RL CQL notebook 
  - Offline RL framing (state/action/reward), episode construction, CQL training, and policy evaluation (Avg Q, approval rate, expected return, baseline return).



# How to Run This Project on Google Colab (Dataset NOT included in repo)

Because the dataset is large, it is **not included** in this repository.  
Follow the steps below to run all notebooks fully on Google Colab.

---

## STEP 1 — Open the Notebook in Google Colab
1. Go to the `notebooks/` folder in this repo.
2. Open any notebook (01_EDA.ipynb, 02_TabNet.ipynb, 03_CQL.ipynb).
3. Click the **"Open in Colab"** button at the top.
   - If the button doesn’t appear, copy the notebook URL and paste it into:  
     https://colab.research.google.com/

---

## STEP 2 — Install Required Libraries in Colab
Run this cell at the top of the notebook:

python
!pip install pytorch-tabnet d3rlpy
!pip install torch numpy pandas scikit-learn matplotlib seaborn tqdm

These commands install TabNet, CQL (d3rlpy), and all ML dependencies.

## STEP 3 — Upload Your Dataset Manually in Colab

Since the dataset is not stored in the repository, upload it manually each time:

from google.colab import files
uploaded = files.upload()


Then choose your dataset file (example: loan_data.csv).

Load it by running:

import pandas as pd
df = pd.read_csv("loan_data.csv")

## Run the Notebooks in Order

To reproduce all results correctly, run the notebooks in this exact order:

1️⃣ Preprocessing_and_EDA.ipynb

Cleans data

Encodes features

Creates target variable

2️⃣ TabNet_DL_Model.ipynb

Trains the TabNet classifier

Computes AUC and F1-score

Tunes optimal decision threshold

3️⃣ RL_Model.ipynb

Converts dataset → RL episodes

Trains CQL (Conservative Q-Learning)

Computes:

Average Q-value

Approval rate

Estimated return

Baseline return

Compares TabNet vs RL decisions
