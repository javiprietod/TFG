# TFG: Counterfactual Explanations for Banking Neural Networks

This repository implements a methodology for generating **counterfactual explanations** of differentiable classifiers (neural nets, logistic regression) applied to banking use-cases such as loan default prediction. Given a “black-box” model and a single input, our method finds the minimal, actionable changes to flip its decision.


## 🚀 Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/javiprietod/TFG.git
   cd TFG
   ```
2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🔧 Configuration

All dataset metadata (paths, target column names, good\_class label, immutable features, weight masks, etc.) live in **datasets.yaml**.
Models and training hyperparameters are in **train.yaml**.

---

## 📊 Data

Three sample banking datasets are included:

* **Loan\_default.csv** — Public “loan default” dataset (Kaggle).
* **santander.csv** — Santander customer default data.
* **spambase.csv** — UCI “spam” dataset (tabular features).

Each CSV expects a binary target (0/1) column, with 1 = “good” (non-default) by default.

---

## 🏋️‍♂️ Training

Train a neural‐network or logistic model:

```bash
python -m src.train data/Loan_default.csv
```

This uses hyperparameters from **train.yaml**, logs to TensorBoard under `runs/`, and saves a TorchScript model.

---

## 🔍 Counterfactual Explanations

Generate counterfactuals (minimal actionable changes) for a single instance:

```bash
python -m src.counterfactual --data data/Loan_default.csv --index 42 --model runs/model_small
```

Or launch the Streamlit interface:

```bash
streamlit run interface.py
```

Follow the web form to upload an instance, adjust feature-change weights, and view the recommended changes.

---

## 📑 Documentation

* **documentacion/tfg.pdf** — Final project write-up.
* **documentacion/images/** — Diagrams, process charts, website screenshots.
* **notebooks/** — Exploratory analyses (gradient‐based counterfactual, logistic demo).

---

## 📄 License

This project is released under the MIT License. See **LICENSE** for details.
Feel free to adapt and extend for your own banking explainability needs!
