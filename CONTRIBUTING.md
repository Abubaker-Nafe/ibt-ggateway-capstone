# Contributing Guide

Thanks for contributing to this capstone project! This repo is organized for a Google Colab workflow.

## Team Workflow (Colab)
We are 5 teammates. Each teammate works on their own notebook in Colab, then we upload/merge into this repo.

### Rules (important)
- **Do not edit someone else’s notebook file** unless agreed.
- Keep notebooks **small and focused** (EDA separate from modeling, etc.).
- Before uploading, ensure the notebook **runs top-to-bottom** without errors.

## Folder Structure
- `notebooks/` → all Colab notebooks (`.ipynb`)
- `data/` → **NO raw big data**. Only small samples and `data/README.md`
- `reports/` → images, charts, exported tables, results (if needed)
- `slide-deck/` → final presentation materials

## Notebook Naming Convention
Use this format:
`NN_topic_owner.ipynb`

Examples:
- `01_eda_nafe.ipynb`
- `02_cleaning_Afaf.ipynb`
- `03_baseline_model_Ayham.ipynb`
- `04_feature_engineering_Tala.ipynb`
- `05_final_model_Abood.ipynb`

Notes:
- `NN` is the order (01, 02, 03…)
- `topic` is short and clear
- `owner` is your first name (if needed)

## Upload / Handoff Process
1. In Colab, run **Runtime → Run all** to confirm it works.
2. Save the notebook with the correct name.
3. Upload the Notebook:
   - the `.ipynb` file (download from Colab) **OR**
   - a Colab link with permission set to **Anyone with the link can view**
   - Upload it to `/notebooks/` and write a short commit message: `Add 01_eda_nafe notebook` 
4. If the notebook produces outputs (plots/tables), we can export key ones to `/reports/`.

## Data Rules
- Do **not** commit large datasets.
- If using a dataset:
  - add the dataset source link to `data/README.md`
  - document how to download/access it
- If you must store a tiny sample dataset, keep it under ~5–10 MB.

## Code Quality (simple checklist)
- Add a title + short description at the top of the notebook
- Clear section headers:
  - Data loading
  - Cleaning
  - EDA
  - Feature engineering
  - Modeling
  - Evaluation
  - Conclusion / Next steps
