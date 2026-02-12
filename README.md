# 🎯 Trader Performance vs Market Sentiment

**Primetrade.ai — Data Science Intern Assignment**

Analyzing how Bitcoin market sentiment (Fear/Greed) relates to trader behavior and performance on Hyperliquid.

---

## 📁 Project Structure

```
primetrade-assignment/
├── data/
│   ├── raw/                  # Original datasets
│   │   ├── fear_greed_index.csv
│   │   └── historical_data.csv
│   └── processed/            # Cleaned & merged data
├── notebooks/
│   ├── analysis.py           # Main analysis (Jupytext format)
│   └── analysis.ipynb        # Jupyter notebook (auto-generated)
├── outputs/
│   ├── charts/               # All visualizations (PNG)
│   └── tables/               # Summary tables (CSV)
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the analysis
```bash
cd notebooks
jupyter notebook analysis.ipynb
```

Or run as a script:
```bash
cd notebooks
python analysis.py
```

---

## 📊 Methodology

### Part A — Data Preparation
- Loaded 2,645 days of Bitcoin Fear & Greed Index (2018–2025) and 211K+ Hyperliquid trades
- Cleaned timestamps, handled missing values, removed duplicates
- Aligned datasets by date; computed 6 daily metrics: PnL, win rate, trade frequency, long/short ratio, avg trade size, drawdown proxy

### Part B — Analysis
1. **Performance on Fear vs Greed days** — Box plots, violin plots, Mann-Whitney U tests comparing PnL, win rate, and drawdown
2. **Behavioral changes by sentiment** — Trade frequency, position sizing, and directional bias shift analysis
3. **Trader segmentation** — 3 segments: High/Low Size, Frequent/Infrequent, Consistent/Inconsistent — cross-tabulated with sentiment
4. **4+ Insights** — PnL by category, volume response, long/short shifts, sentiment–PnL correlation

### Part C — Actionable Output
Two evidence-backed strategy recommendations:
1. **Sentiment-Aware Position Sizing** — Reduce sizes on Fear days for high-size traders
2. **Selective Trading Frequency** — Lower trade count on Fear days to avoid overtrading

### Bonus
- **Predictive Model** — Logistic Regression, Random Forest, Gradient Boosting predicting next-day profitability
- **Trader Clustering** — K-Means with PCA visualization identifying behavioral archetypes

---

## 📈 Key Insights

1. **PnL differs by sentiment regime** — Statistically significant performance gaps between Fear and Greed days
2. **Traders adjust behavior** — Trade frequency, sizing, and directional bias shift with sentiment
3. **Long/Short bias tracks sentiment** — Higher long ratios during Greed, different dynamics during Fear
4. **Lagged features + sentiment provide predictive signal** — Moderate AUC for next-day profitability prediction

---

## 👤 Author

Data Science Intern Candidate — Primetrade.ai Assignment
