# Credit Scoring with Machine Learning
### Predictive Models for Financial Risk Assessment

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Author:** Lorenzo Federico Lai  
> **Affiliation:** POLIMI Graduate School of Management (GSoM) — International Master in Digital Innovation & New Business Design  
> **Related thesis:** *"Machine Learning e Credit Scoring: Modelli Predittivi per la Valutazione del Rischio Finanziario"* — University of Cagliari, A.A. 2023/2024  
> **Supervisor:** Prof. Fabrizio Crespi

---

## Overview

This project implements and compares multiple machine learning classification models for **credit default prediction**, translating the theoretical framework of my bachelor's thesis into working code.

The methodology is grounded in:
- **Banca d'Italia Occasional Paper No. 721** (2022) — *Intelligenza artificiale nel credit scoring*
- **Banca d'Italia Occasional Paper No. 674** (2022) — *Explainable Artificial Intelligence*
- **EBA guidelines** on AI governance in credit risk assessment
- **Alonso & Carbò (2020)** — ML accuracy improvements of 2–10pp over traditional econometric models

---

## Models Implemented

| Model | Type | Explainability | Notes |
|-------|------|----------------|-------|
| **Logistic Regression** | Linear | ★★★★★ | Baseline econometric model (traditional approach) |
| **Decision Tree** | Non-linear | ★★★★☆ | High explainability — preferred by regulators |
| **Random Forest** | Ensemble (Bagging) | ★★★☆☆ | Best accuracy–explainability trade-off |
| **Gradient Boosting** | Ensemble (Boosting) | ★★☆☆☆ | Highest predictive power |

Ensemble methods reflect the industry trend documented in the Banca d'Italia paper: most real-world credit scoring models use **gradient boosting trees** as they balance accuracy and partial explainability through post-hoc techniques.

---

## Explainable AI (XAI)

The project implements **feature importance analysis** as a proxy for Shapley values, consistent with the XAI methodology discussed in the thesis (Section 3.3).

> *"The Shapley value is inspired by game theory and helps us understand the marginal contribution of each variable to the final prediction."*

This approach aligns with EBA requirements that credit scoring models must:
- Be understandable in their operation
- Include governance systems to prevent bias
- Produce verifiable and documented outputs

---

## Fairness Analysis

Section 2.4 of the thesis discusses cognitive biases in credit scoring, citing a **Journal of Financial Economics** study showing that minority borrowers faced 14% higher rejection rates and 8% higher interest rates despite identical credit profiles.

This project includes a **statistical parity analysis** across:
- Gender groups
- Education levels
- Age cohorts

---

## Results

| Model | AUC | CV AUC | Recall (default) |
|-------|-----|--------|-----------------|
| Logistic Regression | 0.630 | 0.624 ± 0.009 | 0.572 |
| Random Forest | 0.628 | 0.625 ± 0.009 | 0.418 |
| Gradient Boosting | 0.618 | 0.618 ± 0.009 | 0.015 |
| Decision Tree | 0.601 | 0.603 ± 0.010 | 0.544 |

> **Note:** Results shown are on synthetic data (UCI-compatible schema). Running with the real [UCI Default of Credit Card Clients dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) typically yields AUC of 0.76–0.82 for Random Forest.

---

## Output Visualisations

| File | Description |
|------|-------------|
| `01_eda.png` | Exploratory data analysis — distributions, default rates by demographic |
| `02_model_comparison.png` | ROC curves, AUC comparison with CV error bars, metrics heatmap |
| `03_feature_importance_xai.png` | Feature importance per model (XAI proxy for Shapley values) |
| `04_fairness_analysis.png` | Statistical parity across gender, education, and age groups |

---

## Installation & Usage

```bash
# Clone the repository
git clone https://github.com/EmilioLussu/credit.scoring.ml.git
cd credit-scoring-ml

# Install dependencies
pip install -r requirements.txt

# Run the full analysis
python credit_scoring_analysis.py
```

Outputs are saved to the `outputs/` directory.

---

## Project Structure

```
credit-scoring-ml/
├── credit_scoring_analysis.py   # Main analysis script
├── requirements.txt             # Dependencies
├── README.md                    # This file
└── outputs/                     # Generated plots (auto-created)
    ├── 01_eda.png
    ├── 02_model_comparison.png
    ├── 03_feature_importance_xai.png
    └── 04_fairness_analysis.png
```

---

## Key References

- Bonaccorsi di Patti E. et al. (2022). *Intelligenza artificiale nel credit scoring*. Banca d'Italia Occasional Paper, 721.
- Cascarino G. et al. (2022). *Explainable Artificial Intelligence: interpreting default forecasting model based on Machine Learning*. Banca d'Italia Occasional Paper, 674.
- Alonso & Carbò (2020). Machine learning in credit risk: bias, accuracy and explainability.
- Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications, 36(2), 2473–2480.
- EU AI Act — Regulation (EU) 2024/1689, European Parliament.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*This project was developed as a practical implementation of the theoretical framework explored in my bachelor's thesis on ML-based credit scoring, subsequently extended during my Master's in Digital Innovation at POLIMI GSoM.*
