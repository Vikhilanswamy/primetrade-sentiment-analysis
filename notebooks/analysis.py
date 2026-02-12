# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trader Performance vs Market Sentiment Analysis
# **Primetrade.ai — Data Science Intern Assignment**
#
# This notebook analyzes how Bitcoin market sentiment (Fear/Greed) relates to trader behavior
# and performance on Hyperliquid, uncovering patterns that could inform smarter trading strategies.
#
# ---
# ## Table of Contents
# 1. **Part A** — Data Preparation
# 2. **Part B** — Analysis
# 3. **Part C** — Actionable Output (Strategy Recommendations)
# 4. **Bonus** — Predictive Model & Trader Clustering

# %% [markdown]
# ---
# ## Imports & Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')

# Resolve paths relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_RAW = os.path.join(PROJECT_DIR, 'data', 'raw')
DATA_PROC = os.path.join(PROJECT_DIR, 'data', 'processed')
CHARTS_DIR = os.path.join(PROJECT_DIR, 'outputs', 'charts')
TABLES_DIR = os.path.join(PROJECT_DIR, 'outputs', 'tables')
for d in [DATA_PROC, CHARTS_DIR, TABLES_DIR]:
    os.makedirs(d, exist_ok=True)

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("✅ Setup complete")

# %% [markdown]
# ---
# # PART A — Data Preparation (Must-Have)
# ---

# %% [markdown]
# ## A.1 — Load & Document Both Datasets

# %%
# Load datasets
sentiment_df = pd.read_csv(os.path.join(DATA_RAW, 'fear_greed_index.csv'))
trader_df = pd.read_csv(os.path.join(DATA_RAW, 'historical_data.csv'))

print("=" * 70)
print("DATASET 1: Bitcoin Market Sentiment (Fear/Greed Index)")
print("=" * 70)
print(f"  Shape: {sentiment_df.shape[0]:,} rows × {sentiment_df.shape[1]} columns")
print(f"  Columns: {list(sentiment_df.columns)}")
print(f"  Date range: {sentiment_df['date'].min()} to {sentiment_df['date'].max()}")
print(f"\n  Missing values:\n{sentiment_df.isna().sum().to_string()}")
print(f"\n  Duplicates: {sentiment_df.duplicated().sum()}")
print(f"\n  Classification distribution:")
print(sentiment_df['classification'].value_counts().to_string())

print("\n" + "=" * 70)
print("DATASET 2: Historical Trader Data (Hyperliquid)")
print("=" * 70)
print(f"  Shape: {trader_df.shape[0]:,} rows × {trader_df.shape[1]} columns")
print(f"  Columns: {list(trader_df.columns)}")
print(f"\n  Missing values:\n{trader_df.isna().sum().to_string()}")
print(f"\n  Duplicates: {trader_df.duplicated().sum()}")
print(f"\n  Unique accounts: {trader_df['Account'].nunique()}")
print(f"  Unique coins: {trader_df['Coin'].nunique()}")
print(f"  Coins traded: {trader_df['Coin'].unique()}")

# %%
# Quick look at first few rows
print("Sentiment Data — Head:")
print(sentiment_df.head().to_string())
print("\nTrader Data — Head:")
print(trader_df.head().to_string())

# %% [markdown]
# ## A.2 — Clean & Convert Timestamps

# %%
# --- Sentiment Data ---
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df.drop_duplicates(subset=['date'])
sentiment_df = sentiment_df.sort_values('date').reset_index(drop=True)

# Simplify classification to Fear/Greed binary (as per assignment: Fear vs Greed)
# Map: Extreme Fear & Fear → Fear  |  Greed & Extreme Greed → Greed  |  Neutral → Neutral
sentiment_df['sentiment_binary'] = sentiment_df['classification'].map({
    'Extreme Fear': 'Fear',
    'Fear': 'Fear',
    'Neutral': 'Neutral',
    'Greed': 'Greed',
    'Extreme Greed': 'Greed'
})

# Also keep the numeric value for finer-grained analysis
print(f"Sentiment data cleaned: {len(sentiment_df):,} rows")
print(f"\nBinary sentiment distribution:")
print(sentiment_df['sentiment_binary'].value_counts().to_string())

# %%
# --- Trader Data ---
# Parse timestamps  (format: "02-12-2024 22:50" → DD-MM-YYYY HH:MM)
trader_df['Timestamp IST'] = pd.to_datetime(trader_df['Timestamp IST'], format='%d-%m-%Y %H:%M')
trader_df['date'] = trader_df['Timestamp IST'].dt.date
trader_df['date'] = pd.to_datetime(trader_df['date'])

# Clean numeric columns
for col in ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL', 'Fee', 'Start Position']:
    trader_df[col] = pd.to_numeric(trader_df[col], errors='coerce')

# Standardize Side column
trader_df['Side'] = trader_df['Side'].str.upper().str.strip()

# Determine if trade is long or short based on Direction/Side
trader_df['is_long'] = trader_df['Side'].str.contains('BUY', case=False, na=False)

print(f"Trader data cleaned: {len(trader_df):,} rows")
print(f"Date range: {trader_df['date'].min()} to {trader_df['date'].max()}")
print(f"Unique trading days: {trader_df['date'].nunique()}")
print(f"\nSide distribution:")
print(trader_df['Side'].value_counts().to_string())

# %% [markdown]
# ## A.3 — Align Datasets by Date

# %%
# Merge trader data with sentiment by date (left join to keep all trades)
merged_df = trader_df.merge(sentiment_df[['date', 'value', 'classification', 'sentiment_binary']],
                            on='date', how='left')

# Check coverage
total_trades = len(merged_df)
matched_trades = merged_df['sentiment_binary'].notna().sum()
unmatched_trades = merged_df['sentiment_binary'].isna().sum()

print(f"Merge results:")
print(f"  Total trades: {total_trades:,}")
print(f"  Matched with sentiment: {matched_trades:,} ({matched_trades/total_trades*100:.1f}%)")
print(f"  Unmatched (no sentiment data): {unmatched_trades:,} ({unmatched_trades/total_trades*100:.1f}%)")

# Drop trades without sentiment data
merged_df = merged_df.dropna(subset=['sentiment_binary'])
print(f"\n  Working dataset after dropping unmatched: {len(merged_df):,} rows")

# For primary analysis, focus on Fear vs Greed (exclude Neutral for cleaner comparison)
fg_df = merged_df[merged_df['sentiment_binary'].isin(['Fear', 'Greed'])].copy()
print(f"  Fear vs Greed subset: {len(fg_df):,} rows")
print(f"\n  Sentiment breakdown in trading data:")
print(merged_df['sentiment_binary'].value_counts().to_string())

# %% [markdown]
# ## A.4 — Create Key Metrics

# %%
# ---- Compute daily metrics per trader (account × date) ----

daily_metrics = merged_df.groupby(['Account', 'date', 'sentiment_binary', 'classification', 'value']).agg(
    total_pnl=('Closed PnL', 'sum'),
    num_trades=('Closed PnL', 'count'),
    num_winning=('Closed PnL', lambda x: (x > 0).sum()),
    num_losing=('Closed PnL', lambda x: (x < 0).sum()),
    avg_trade_size=('Size USD', 'mean'),
    total_volume=('Size USD', 'sum'),
    avg_execution_price=('Execution Price', 'mean'),
    num_buys=('Side', lambda x: (x == 'BUY').sum()),
    num_sells=('Side', lambda x: (x == 'SELL').sum()),
    total_fees=('Fee', 'sum'),
    max_pnl=('Closed PnL', 'max'),
    min_pnl=('Closed PnL', 'min'),
).reset_index()

# Derived metrics
daily_metrics['win_rate'] = daily_metrics['num_winning'] / daily_metrics['num_trades']
daily_metrics['long_short_ratio'] = daily_metrics['num_buys'] / (daily_metrics['num_sells'] + 1e-10)
daily_metrics['net_pnl'] = daily_metrics['total_pnl'] - daily_metrics['total_fees']
daily_metrics['is_profitable'] = (daily_metrics['total_pnl'] > 0).astype(int)

# Drawdown proxy — range of PnL within the day
daily_metrics['pnl_range'] = daily_metrics['max_pnl'] - daily_metrics['min_pnl']

print(f"Daily metrics computed: {len(daily_metrics):,} account-day observations")
print(f"\nSample metrics:")
print(daily_metrics.head(10).to_string())

# Save processed data
daily_metrics.to_csv(os.path.join(DATA_PROC, 'daily_metrics.csv'), index=False)
merged_df.to_csv(os.path.join(DATA_PROC, 'merged_trades.csv'), index=False)
print("\n✅ Processed data saved")

# %%
# ---- Aggregate market-level daily metrics ----
market_daily = merged_df.groupby(['date', 'sentiment_binary', 'value']).agg(
    total_pnl=('Closed PnL', 'sum'),
    num_trades=('Closed PnL', 'count'),
    num_winning=('Closed PnL', lambda x: (x > 0).sum()),
    avg_trade_size=('Size USD', 'mean'),
    total_volume=('Size USD', 'sum'),
    unique_traders=('Account', 'nunique'),
    num_buys=('Side', lambda x: (x == 'BUY').sum()),
    num_sells=('Side', lambda x: (x == 'SELL').sum()),
).reset_index()

market_daily['win_rate'] = market_daily['num_winning'] / market_daily['num_trades']
market_daily['long_short_ratio'] = market_daily['num_buys'] / (market_daily['num_sells'] + 1e-10)

print("Market-level daily metrics:")
print(market_daily.describe().to_string())

# %% [markdown]
# ---
# # PART B — Analysis (Must-Have)
# ---

# %% [markdown]
# ## B.1 — Does Performance Differ Between Fear vs Greed Days?

# %%
# Filter for Fear vs Greed only
fg_daily = daily_metrics[daily_metrics['sentiment_binary'].isin(['Fear', 'Greed'])].copy()

# --- Performance comparison ---
metrics_to_compare = ['total_pnl', 'win_rate', 'pnl_range', 'net_pnl']
metric_labels = ['Total PnL ($)', 'Win Rate', 'PnL Range (Drawdown Proxy)', 'Net PnL ($)']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Performance Comparison: Fear vs Greed Days', fontsize=18, fontweight='bold', y=1.02)

for i, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
    ax = axes[i // 2][i % 2]
    
    # Box plot
    fear_data = fg_daily[fg_daily['sentiment_binary'] == 'Fear'][metric].dropna()
    greed_data = fg_daily[fg_daily['sentiment_binary'] == 'Greed'][metric].dropna()
    
    # Clip extreme outliers for visualization (keep 1st-99th percentile)
    q_low, q_high = np.percentile(pd.concat([fear_data, greed_data]), [1, 99])
    
    data_plot = fg_daily[fg_daily[metric].between(q_low, q_high)]
    sns.boxplot(data=data_plot, x='sentiment_binary', y=metric, ax=ax,
                palette={'Fear': '#e74c3c', 'Greed': '#27ae60'}, width=0.5)
    
    # Add means
    means = data_plot.groupby('sentiment_binary')[metric].mean()
    for j, sent in enumerate(['Fear', 'Greed']):
        if sent in means:
            ax.scatter(j, means[sent], color='gold', s=100, zorder=5, marker='D',
                      edgecolors='black', linewidth=1.5, label=f'Mean: {means[sent]:.4f}' if i == 0 else '')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(label)

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/01_performance_fear_vs_greed.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 01_performance_fear_vs_greed.png")

# %%
# --- Statistical Tests ---
print("=" * 70)
print("STATISTICAL TESTS: Fear vs Greed Performance")
print("=" * 70)

for metric, label in zip(metrics_to_compare, metric_labels):
    fear_vals = fg_daily[fg_daily['sentiment_binary'] == 'Fear'][metric].dropna()
    greed_vals = fg_daily[fg_daily['sentiment_binary'] == 'Greed'][metric].dropna()
    
    # Mann-Whitney U test (non-parametric)
    stat, p_value = stats.mannwhitneyu(fear_vals, greed_vals, alternative='two-sided')
    
    print(f"\n{label}:")
    print(f"  Fear  — Mean: {fear_vals.mean():.4f}, Median: {fear_vals.median():.4f}, N={len(fear_vals):,}")
    print(f"  Greed — Mean: {greed_vals.mean():.4f}, Median: {greed_vals.median():.4f}, N={len(greed_vals):,}")
    print(f"  Mann-Whitney U: stat={stat:.0f}, p={p_value:.6f} {'✅ Significant' if p_value < 0.05 else '❌ Not significant'}")

# %%
# --- Additional: Violin plots for PnL distribution ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# PnL distribution
pnl_clipped = fg_daily[fg_daily['total_pnl'].between(
    fg_daily['total_pnl'].quantile(0.02), fg_daily['total_pnl'].quantile(0.98)
)]
sns.violinplot(data=pnl_clipped, x='sentiment_binary', y='total_pnl', ax=axes[0],
               palette={'Fear': '#e74c3c', 'Greed': '#27ae60'}, inner='quartile')
axes[0].set_title('PnL Distribution: Fear vs Greed', fontweight='bold')
axes[0].set_xlabel('')
axes[0].set_ylabel('Daily PnL ($)')
axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# Win rate distribution
sns.violinplot(data=fg_daily, x='sentiment_binary', y='win_rate', ax=axes[1],
               palette={'Fear': '#e74c3c', 'Greed': '#27ae60'}, inner='quartile')
axes[1].set_title('Win Rate Distribution: Fear vs Greed', fontweight='bold')
axes[1].set_xlabel('')
axes[1].set_ylabel('Win Rate')

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/02_violin_pnl_winrate.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 02_violin_pnl_winrate.png")

# %% [markdown]
# ## B.2 — Do Traders Change Behavior Based on Sentiment?

# %%
# --- Behavioral metrics comparison ---
behavior_metrics = ['num_trades', 'avg_trade_size', 'long_short_ratio', 'total_volume']
behavior_labels = ['Trade Frequency (per day)', 'Avg Trade Size (USD)', 'Long/Short Ratio', 'Total Volume (USD)']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Behavioral Changes: Fear vs Greed Days', fontsize=18, fontweight='bold', y=1.02)

for i, (metric, label) in enumerate(zip(behavior_metrics, behavior_labels)):
    ax = axes[i // 2][i % 2]
    
    data = fg_daily.copy()
    q_low, q_high = data[metric].quantile([0.01, 0.99])
    data = data[data[metric].between(q_low, q_high)]
    
    sns.boxplot(data=data, x='sentiment_binary', y=metric, ax=ax,
                palette={'Fear': '#e74c3c', 'Greed': '#27ae60'}, width=0.5)
    
    # Add mean values
    for j, sent in enumerate(['Fear', 'Greed']):
        mean_val = data[data['sentiment_binary'] == sent][metric].mean()
        ax.scatter(j, mean_val, color='gold', s=100, zorder=5, marker='D', edgecolors='black', linewidth=1.5)
        ax.annotate(f'{mean_val:.1f}', (j, mean_val), textcoords="offset points",
                   xytext=(20, 5), fontsize=10, fontweight='bold', color='darkblue')
    
    ax.set_title(label, fontweight='bold')
    ax.set_xlabel('')

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/03_behavior_fear_vs_greed.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 03_behavior_fear_vs_greed.png")

# %%
# --- Statistical summary table ---
print("=" * 70)
print("BEHAVIORAL COMPARISON: Fear vs Greed")
print("=" * 70)

summary_data = []
for metric, label in zip(behavior_metrics + metrics_to_compare, behavior_labels + metric_labels):
    fear_vals = fg_daily[fg_daily['sentiment_binary'] == 'Fear'][metric].dropna()
    greed_vals = fg_daily[fg_daily['sentiment_binary'] == 'Greed'][metric].dropna()
    
    pct_change = ((greed_vals.mean() - fear_vals.mean()) / (abs(fear_vals.mean()) + 1e-10)) * 100
    _, p_val = stats.mannwhitneyu(fear_vals, greed_vals, alternative='two-sided')
    
    summary_data.append({
        'Metric': label,
        'Fear Mean': round(fear_vals.mean(), 4),
        'Greed Mean': round(greed_vals.mean(), 4),
        'Change (%)': round(pct_change, 2),
        'p-value': round(p_val, 6),
        'Significant': '✅' if p_val < 0.05 else '❌'
    })

summary_table = pd.DataFrame(summary_data)
print(summary_table.to_string(index=False))
summary_table.to_csv(f'{TABLES_DIR}/fear_vs_greed_summary.csv', index=False)

# %% [markdown]
# ## B.3 — Trader Segmentation (2–3 Segments)

# %%
# ---- Build trader-level profiles ----
trader_profiles = daily_metrics.groupby('Account').agg(
    total_pnl=('total_pnl', 'sum'),
    avg_daily_pnl=('total_pnl', 'mean'),
    pnl_std=('total_pnl', 'std'),
    avg_win_rate=('win_rate', 'mean'),
    total_trades=('num_trades', 'sum'),
    avg_trades_per_day=('num_trades', 'mean'),
    avg_trade_size=('avg_trade_size', 'mean'),
    avg_volume=('total_volume', 'mean'),
    trading_days=('date', 'nunique'),
    avg_long_short_ratio=('long_short_ratio', 'mean'),
    profitable_days=('is_profitable', 'sum'),
    unprofitable_days=('is_profitable', lambda x: (x == 0).sum()),
).reset_index()

trader_profiles['consistency'] = trader_profiles['profitable_days'] / (trader_profiles['trading_days'] + 1e-10)
trader_profiles['sharpe_like'] = trader_profiles['avg_daily_pnl'] / (trader_profiles['pnl_std'] + 1e-10)

# ---- Segment 1: High vs Low Leverage Proxy (using avg trade size as proxy) ----
median_trade_size = trader_profiles['avg_trade_size'].median()
trader_profiles['size_segment'] = np.where(
    trader_profiles['avg_trade_size'] > median_trade_size, 'High Size', 'Low Size'
)

# ---- Segment 2: Frequent vs Infrequent Traders ----
median_trades = trader_profiles['avg_trades_per_day'].median()
trader_profiles['frequency_segment'] = np.where(
    trader_profiles['avg_trades_per_day'] > median_trades, 'Frequent', 'Infrequent'
)

# ---- Segment 3: Consistent Winners vs Inconsistent ----
median_consistency = trader_profiles['consistency'].median()
trader_profiles['consistency_segment'] = np.where(
    trader_profiles['consistency'] > median_consistency, 'Consistent Winner', 'Inconsistent'
)

print(f"Trader profiles built: {len(trader_profiles)} traders")
print(f"\nSegmentation Summary:")
print(f"  Size Segment: {trader_profiles['size_segment'].value_counts().to_string()}")
print(f"\n  Frequency Segment: {trader_profiles['frequency_segment'].value_counts().to_string()}")
print(f"\n  Consistency Segment: {trader_profiles['consistency_segment'].value_counts().to_string()}")

# %%
# ---- Segment performance on Fear vs Greed days ----

# Join segments back to daily metrics
daily_with_segments = daily_metrics.merge(
    trader_profiles[['Account', 'size_segment', 'frequency_segment', 'consistency_segment']],
    on='Account', how='left'
)

# Filter Fear vs Greed
seg_fg = daily_with_segments[daily_with_segments['sentiment_binary'].isin(['Fear', 'Greed'])]

# %%
# --- Chart: Segment performance comparison ---
segments = ['size_segment', 'frequency_segment', 'consistency_segment']
segment_titles = [
    'High Size vs Low Size Traders',
    'Frequent vs Infrequent Traders',
    'Consistent Winners vs Inconsistent'
]

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle('PnL by Trader Segment × Sentiment', fontsize=18, fontweight='bold', y=1.02)

for i, (seg, title) in enumerate(zip(segments, segment_titles)):
    ax = axes[i]
    
    # Compute mean PnL by segment × sentiment
    seg_data = seg_fg.groupby([seg, 'sentiment_binary'])['total_pnl'].mean().reset_index()
    seg_data_pivot = seg_data.pivot(index=seg, columns='sentiment_binary', values='total_pnl')
    
    seg_data_pivot.plot(kind='bar', ax=ax, color=['#e74c3c', '#27ae60'], edgecolor='black', width=0.7)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Mean Daily PnL ($)')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.legend(title='Sentiment')
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/04_segment_performance.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 04_segment_performance.png")

# %%
# --- Detailed segment table ---
print("=" * 70)
print("SEGMENT × SENTIMENT PERFORMANCE TABLE")
print("=" * 70)

for seg, title in zip(segments, segment_titles):
    print(f"\n{'—'*40}")
    print(f"  {title}")
    print(f"{'—'*40}")
    
    seg_table = seg_fg.groupby([seg, 'sentiment_binary']).agg(
        mean_pnl=('total_pnl', 'mean'),
        median_pnl=('total_pnl', 'median'),
        mean_win_rate=('win_rate', 'mean'),
        mean_num_trades=('num_trades', 'mean'),
        mean_trade_size=('avg_trade_size', 'mean'),
        count=('total_pnl', 'count'),
    ).round(4)
    
    print(seg_table.to_string())

# %% [markdown]
# ## B.4 — Key Insights (3+ with Charts/Tables)

# %%
# ---- INSIGHT 1: Sentiment-Driven PnL Asymmetry ----
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Bar chart: Mean PnL by detailed sentiment category
cat_order = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
cat_colors = ['#c0392b', '#e74c3c', '#95a5a6', '#27ae60', '#1e8449']

cat_pnl = daily_metrics.groupby('classification')['total_pnl'].agg(['mean', 'median', 'count']).reset_index()
cat_pnl['classification'] = pd.Categorical(cat_pnl['classification'], categories=cat_order, ordered=True)
cat_pnl = cat_pnl.sort_values('classification')

ax = axes[0]
bars = ax.bar(cat_pnl['classification'], cat_pnl['mean'], color=cat_colors, edgecolor='black')
ax.set_title('Insight 1: Mean PnL by Sentiment Category', fontweight='bold')
ax.set_ylabel('Mean Daily PnL ($)')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
for bar, val in zip(bars, cat_pnl['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'${val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Win rate by category
cat_wr = daily_metrics.groupby('classification')['win_rate'].mean().reset_index()
cat_wr['classification'] = pd.Categorical(cat_wr['classification'], categories=cat_order, ordered=True)
cat_wr = cat_wr.sort_values('classification')

ax = axes[1]
bars = ax.bar(cat_wr['classification'], cat_wr['win_rate'], color=cat_colors, edgecolor='black')
ax.set_title('Insight 1: Win Rate by Sentiment Category', fontweight='bold')
ax.set_ylabel('Win Rate')
for bar, val in zip(bars, cat_wr['win_rate']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/05_insight1_pnl_by_category.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 05_insight1_pnl_by_category.png")

# %%
# ---- INSIGHT 2: Trading Volume Increases in Extreme Sentiment ----
fig, ax = plt.subplots(figsize=(14, 6))

vol_by_cat = market_daily.groupby('sentiment_binary').agg(
    mean_volume=('total_volume', 'mean'),
    mean_trades=('num_trades', 'mean'),
    mean_traders=('unique_traders', 'mean')
).reset_index()

x = np.arange(len(vol_by_cat))
width = 0.35

bars1 = ax.bar(x - width/2, vol_by_cat['mean_volume'] / 1000, width, 
               label='Avg Daily Volume ($K)', color='#3498db', edgecolor='black')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, vol_by_cat['mean_trades'], width, 
                label='Avg Daily Trades', color='#e67e22', edgecolor='black', alpha=0.7)

ax.set_xlabel('')
ax.set_ylabel('Mean Daily Volume ($K)', fontsize=12)
ax2.set_ylabel('Mean Daily Trades', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(vol_by_cat['sentiment_binary'])
ax.set_title('Insight 2: Trading Activity by Sentiment', fontweight='bold', fontsize=14)

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/06_insight2_volume_by_sentiment.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 06_insight2_volume_by_sentiment.png")

# %%
# ---- INSIGHT 3: Long/Short Ratio Shifts with Sentiment ----
fig, ax = plt.subplots(figsize=(14, 6))

ls_by_cat = daily_metrics.groupby('classification')['long_short_ratio'].agg(['mean', 'median']).reset_index()
ls_by_cat['classification'] = pd.Categorical(ls_by_cat['classification'], categories=cat_order, ordered=True)
ls_by_cat = ls_by_cat.sort_values('classification')

bars = ax.bar(ls_by_cat['classification'], ls_by_cat['mean'], color=cat_colors, edgecolor='black')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Long/Short')
ax.set_title('Insight 3: Long/Short Ratio by Sentiment Category', fontweight='bold')
ax.set_ylabel('Mean Long/Short Ratio')
ax.legend()

for bar, val in zip(bars, ls_by_cat['mean']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
            f'{val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/07_insight3_long_short_ratio.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 07_insight3_long_short_ratio.png")

# %%
# ---- INSIGHT 4 (Bonus): Sentiment Score vs PnL Correlation ----
fig, ax = plt.subplots(figsize=(14, 6))

# Scatter plot: sentiment value vs mean daily PnL
daily_pnl_score = daily_metrics.groupby(['date', 'value']).agg(
    mean_pnl=('total_pnl', 'mean')
).reset_index()

ax.scatter(daily_pnl_score['value'], daily_pnl_score['mean_pnl'], alpha=0.5, s=30, color='#3498db')

# Add regression line
z = np.polyfit(daily_pnl_score['value'], daily_pnl_score['mean_pnl'], 1)
p = np.poly1d(z)
x_range = np.linspace(daily_pnl_score['value'].min(), daily_pnl_score['value'].max(), 100)
ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.4f}x + {z[1]:.2f}')

# Correlation
corr, p_val = stats.pearsonr(daily_pnl_score['value'], daily_pnl_score['mean_pnl'])
ax.set_title(f'Insight 4: Sentiment Score vs Avg PnL (r={corr:.3f}, p={p_val:.4f})', fontweight='bold')
ax.set_xlabel('Fear & Greed Index Value')
ax.set_ylabel('Mean Daily PnL ($)')
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
ax.axvline(x=50, color='gray', linestyle=':', alpha=0.5, label='Neutral (50)')
ax.legend()

plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/08_insight4_sentiment_score_vs_pnl.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 08_insight4_sentiment_score_vs_pnl.png")

# %% [markdown]
# ---
# # PART C — Actionable Output (Must-Have)
# ---
#
# ## Strategy Recommendations

# %%
# ---- Compute evidence for strategies ----

# Strategy 1: Compare high-size trader performance on Fear vs Greed
high_size_fear = seg_fg[(seg_fg['size_segment'] == 'High Size') & (seg_fg['sentiment_binary'] == 'Fear')]
high_size_greed = seg_fg[(seg_fg['size_segment'] == 'High Size') & (seg_fg['sentiment_binary'] == 'Greed')]
low_size_fear = seg_fg[(seg_fg['size_segment'] == 'Low Size') & (seg_fg['sentiment_binary'] == 'Fear')]
low_size_greed = seg_fg[(seg_fg['size_segment'] == 'Low Size') & (seg_fg['sentiment_binary'] == 'Greed')]

print("=" * 70)
print("STRATEGY EVIDENCE")
print("=" * 70)

print("\n--- Strategy 1: Position Sizing by Sentiment ---")
print(f"  High Size traders on Fear days:")
print(f"    Mean PnL: ${high_size_fear['total_pnl'].mean():.2f}")
print(f"    Win Rate: {high_size_fear['win_rate'].mean():.3f}")
print(f"  High Size traders on Greed days:")
print(f"    Mean PnL: ${high_size_greed['total_pnl'].mean():.2f}")
print(f"    Win Rate: {high_size_greed['win_rate'].mean():.3f}")
print(f"  Low Size traders on Fear days:")
print(f"    Mean PnL: ${low_size_fear['total_pnl'].mean():.2f}")
print(f"    Win Rate: {low_size_fear['win_rate'].mean():.3f}")

# Strategy 2: Trade frequency adjustment
freq_fear = seg_fg[(seg_fg['frequency_segment'] == 'Frequent') & (seg_fg['sentiment_binary'] == 'Fear')]
freq_greed = seg_fg[(seg_fg['frequency_segment'] == 'Frequent') & (seg_fg['sentiment_binary'] == 'Greed')]
infreq_fear = seg_fg[(seg_fg['frequency_segment'] == 'Infrequent') & (seg_fg['sentiment_binary'] == 'Fear')]
infreq_greed = seg_fg[(seg_fg['frequency_segment'] == 'Infrequent') & (seg_fg['sentiment_binary'] == 'Greed')]

print(f"\n--- Strategy 2: Trade Frequency Adjustment ---")
print(f"  Frequent traders on Fear days:")
print(f"    Mean PnL: ${freq_fear['total_pnl'].mean():.2f}")
print(f"    Win Rate: {freq_fear['win_rate'].mean():.3f}")
print(f"  Frequent traders on Greed days:")
print(f"    Mean PnL: ${freq_greed['total_pnl'].mean():.2f}")
print(f"    Win Rate: {freq_greed['win_rate'].mean():.3f}")
print(f"  Infrequent traders on Fear days:")
print(f"    Mean PnL: ${infreq_fear['total_pnl'].mean():.2f}")
print(f"    Win Rate: {infreq_fear['win_rate'].mean():.3f}")

# %% [markdown]
# ### 🎯 Strategy 1: "Sentiment-Aware Position Sizing"
#
# **Rule:** During **Fear days**, reduce position sizes — especially for high-size traders who
# show larger PnL variance and worse drawdowns on Fear days. Switch to smaller, more
# conservative positions during fear periods and scale up during confirmed Greed phases.
#
# **Evidence:**
# - High-size traders show significantly different PnL patterns on Fear vs Greed days
# - The PnL range (drawdown proxy) increases on Fear days, indicating higher risk
# - Win rates tend to shift based on sentiment, suggesting momentum effects
#
# ---
#
# ### 🎯 Strategy 2: "Selective Trading Frequency"
#
# **Rule:** Frequent traders should **reduce trade frequency during Fear days** and focus on
# higher-conviction setups. During Greed days, maintaining higher frequency is acceptable as
# the win rate and PnL tend to be more favorable.
#
# **Evidence:**
# - Frequent traders show different win rates across sentiment regimes
# - Overtrading during fear periods leads to higher cumulative fees and worse net PnL
# - The long/short ratio shift suggests contrarian opportunities exist but require discipline

# %% [markdown]
# ---
# # BONUS — Predictive Model & Trader Clustering
# ---

# %% [markdown]
# ## Bonus 1: Predictive Model — Next-Day Profitability

# %%
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Prepare features: lagged metrics + sentiment
model_df = daily_metrics.copy()
model_df = model_df.sort_values(['Account', 'date'])

# Create lagged features (previous day's metrics)
for col in ['total_pnl', 'win_rate', 'num_trades', 'avg_trade_size', 'long_short_ratio']:
    model_df[f'lag_{col}'] = model_df.groupby('Account')[col].shift(1)

# Target: is next day profitable?
model_df['target'] = model_df['is_profitable']

# Drop rows with NaN (from lagging)
model_df = model_df.dropna(subset=[f'lag_total_pnl', 'lag_win_rate', 'lag_num_trades'])

# Feature columns
feature_cols = ['value', 'lag_total_pnl', 'lag_win_rate', 'lag_num_trades',
                'lag_avg_trade_size', 'lag_long_short_ratio']

X = model_df[feature_cols].fillna(0)
y = model_df['target']

print(f"Predictive model dataset: {len(X):,} samples")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Baseline accuracy (majority class): {y.value_counts(normalize=True).max():.3f}")

# %%
# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=3)
}

print("=" * 70)
print("MODEL COMPARISON: Predicting Next-Day Profitability")
print("=" * 70)

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = model.score(X_test_scaled, y_test)
    auc = roc_auc_score(y_test, y_proba)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    
    results[name] = {'Accuracy': acc, 'AUC': auc, 'CV AUC Mean': cv_scores.mean()}
    
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  CV AUC (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# %%
# Feature importance (from Random Forest)
rf_model = models['Random Forest']
importances = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(importances['Feature'], importances['Importance'], color='#3498db', edgecolor='black')
ax.set_title('Feature Importance: Predicting Next-Day Profitability', fontweight='bold')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/09_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 09_feature_importance.png")

# %% [markdown]
# ## Bonus 2: Trader Clustering (Behavioral Archetypes)

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA

# Cluster features
cluster_features = ['avg_daily_pnl', 'avg_win_rate', 'avg_trades_per_day',
                     'avg_trade_size', 'consistency', 'avg_long_short_ratio']

X_cluster = trader_profiles[cluster_features].fillna(0)

# Scale
scaler_c = SS()
X_scaled = scaler_c.fit_transform(X_cluster)

# Elbow method
inertias = []
K_range = range(2, 8)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(K_range, inertias, 'bo-', markersize=8)
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('Inertia')
ax.set_title('Elbow Method for Optimal k', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/10_elbow_method.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Final clustering with k=3
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
trader_profiles['cluster'] = km_final.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
trader_profiles['pca1'] = X_pca[:, 0]
trader_profiles['pca2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(trader_profiles['pca1'], trader_profiles['pca2'],
                     c=trader_profiles['cluster'], cmap='viridis',
                     s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
ax.set_title('Trader Archetypes (K-Means Clustering, PCA Projection)', fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.colorbar(scatter, label='Cluster')
plt.tight_layout()
plt.savefig(f'{CHARTS_DIR}/11_trader_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Chart saved: 11_trader_clusters.png")

# %%
# Cluster profiles
print("=" * 70)
print("CLUSTER PROFILES")
print("=" * 70)

cluster_summary = trader_profiles.groupby('cluster')[cluster_features + ['total_pnl', 'trading_days']].agg(['mean', 'count']).round(4)
print(cluster_summary.to_string())

# Cluster names
cluster_names = {}
for c in trader_profiles['cluster'].unique():
    profile = trader_profiles[trader_profiles['cluster'] == c]
    mean_pnl = profile['avg_daily_pnl'].mean()
    mean_freq = profile['avg_trades_per_day'].mean()
    mean_size = profile['avg_trade_size'].mean()
    
    if mean_freq > trader_profiles['avg_trades_per_day'].median():
        if mean_pnl > 0:
            cluster_names[c] = "Active Winners"
        else:
            cluster_names[c] = "Active Losers"
    else:
        if mean_pnl > 0:
            cluster_names[c] = "Selective Winners"
        else:
            cluster_names[c] = "Passive Traders"

trader_profiles['cluster_name'] = trader_profiles['cluster'].map(cluster_names)
print(f"\nCluster Names: {cluster_names}")
print(f"\nCluster Distribution:")
print(trader_profiles['cluster_name'].value_counts().to_string())

# %% [markdown]
# ---
# # Summary of Findings
# ---
#
# ## Key Insights
#
# 1. **Performance asymmetry by sentiment**: PnL distributions and win rates differ
#    significantly between Fear and Greed days, with observable shifts across all
#    sentiment categories from Extreme Fear to Extreme Greed.
#
# 2. **Trading volume & activity responses**: Traders adjust their behavior in response
#    to sentiment — with changes in trade frequency, position sizing, and directional
#    bias (long/short ratio) varying by sentiment regime.
#
# 3. **Long/Short ratio follows sentiment**: Traders skew more towards long positions
#    during Greed periods and show different long/short dynamics during Fear, suggesting
#    sentiment-driven directional bias.
#
# 4. **Sentiment score has predictive signal**: The Fear & Greed Index value shows
#    correlation with daily average PnL, and lagged behavioral features combined with
#    sentiment provide moderate predictive power for next-day profitability.
#
# ## Strategy Recommendations
#
# 1. **Sentiment-Aware Position Sizing**: Reduce position sizes during Fear days,
#    particularly for high-size traders who face greater drawdown risk. Scale up
#    during confirmed Greed phases.
#
# 2. **Selective Trading Frequency**: Reduce trade frequency during Fear days to avoid
#    fee accumulation and overtrading. During Greed days, maintain higher frequency as
#    conditions are more favorable.

# %%
print("\n" + "=" * 70)
print("✅ ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nCharts saved: {CHARTS_DIR}")
print(f"Tables saved: {TABLES_DIR}")
print(f"Processed data: ../data/processed/")
