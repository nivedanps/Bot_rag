# Website Performance Analysis — Exploratory Data Analysis (EDA)

A professional data analysis project that investigates website traffic and user engagement patterns using Google Analytics export data. The analysis spans multiple marketing channels and time dimensions to surface actionable insights on audience behaviour and content effectiveness.

---

## Project Overview

With over **1.1 billion websites** on the internet as of 2025 and **60%+ of web traffic** originating from mobile devices, understanding how users interact with a website across channels and time is critical for digital growth.

This notebook answers six key business questions through structured EDA and visualisation:

| # | Business Question |
|---|---|
| 1 | What patterns or trends exist in website sessions and users over time? |
| 2 | Which marketing channel brings the highest number of users, and how can other channels be improved? |
| 3 | Which channel has the highest average engagement time, and what does this reveal about content effectiveness? |
| 4 | How does engagement rate vary across different traffic channels? |
| 5 | Which channels drive more engaged sessions vs non-engaged ones, and how can underperforming channels be improved? |
| 6 | At what hours of the day does each channel drive the most traffic? |
| 7 | Is there a correlation between high traffic (sessions) and high engagement rate over time? |

---

## Dataset

| Property | Details |
|---|---|
| File | `data-export (1) (1).csv` |
| Format | CSV (unicode-escaped encoding, Google Analytics export) |
| Source | Google Analytics — channel-level, hourly data |

### Columns

| Column | Description |
|---|---|
| `channel group` | Marketing channel (e.g., Organic Search, Direct, Referral) |
| `DateHour` | Date and hour of traffic (format: `YYYYMMDDHH`) |
| `Users` | Number of unique users |
| `Sessions` | Total sessions initiated |
| `Engaged Session` | Sessions meeting engagement threshold |
| `Average engagement time per session` | Mean active time per session (seconds) |
| `Engaged sessions per user` | Ratio of engaged sessions to users |
| `Events per session` | Average number of events fired per session |
| `Engagement rate` | Proportion of sessions that were engaged |
| `Event count` | Total events recorded |

---

## Libraries Used

| Library | Purpose |
|---|---|
| `numpy` | Numerical aggregation and computations |
| `pandas` | Data ingestion, cleaning, and transformation |
| `matplotlib` | Base plotting and figure configuration |
| `seaborn` | Statistical visualisations (bar, box, heatmap plots) |

---

## Project Structure

```
EDA_website_Project2.ipynb        # Main analysis notebook
data-export (1) (1).csv           # Source dataset (place in /content/)
README_EDA_Website.md             # This file
```

---

## Methodology

### 1. Data Loading
- Read CSV with `encoding='unicode_escape'` to handle special characters in Google Analytics exports
- Promote the first row to column headers using `iloc[0]` indexing
- Rename columns to clean, standardised names

### 2. Data Cleaning & Preprocessing
- Inspect shape, data types (`df.info()`), and summary statistics (`df.describe()`)
- Check and handle null values
- Parse `DateHour` from compact integer format (`YYYYMMDDHH`) to `datetime` using `pd.to_datetime`
- Convert all metric columns to numeric types using `pd.to_numeric(errors='coerce')`
- Extract an `Hour` feature from the parsed datetime for hourly analysis

### 3. Exploratory Data Analysis & Visualisations

#### Q1 — Sessions & Users Over Time
- **Chart:** Line plot (grouped by `DateHour`)
- **Insight:** Identifies traffic spikes, seasonal trends, and growth or decline patterns over the analysis period

#### Q2 — Total Users by Channel
- **Chart:** Bar plot (sum of `Users` per `channel group`)
- **Insight:** Highlights the dominant acquisition channel and opportunities for underperforming channels

#### Q3 — Average Engagement Time by Channel
- **Chart:** Bar plot (mean of `Average engagement time per session` per channel)
- **Insight:** Reveals which channels attract users who actively consume content vs. those who bounce quickly

#### Q4 — Engagement Rate Distribution by Channel
- **Chart:** Box plot (`Engagement rate` per `channel group`, `coolwarm` palette)
- **Insight:** Shows the spread and consistency of engagement across channels, including outliers

#### Q5 — Engaged vs Non-Engaged Sessions by Channel
- **Chart:** Grouped bar plot (melted `Engaged Session` vs computed `Non-Engaged`)
- **Insight:** Pinpoints channels with high non-engagement that may need content or targeting improvements

#### Q6 — Traffic Heatmap by Hour and Channel
- **Chart:** Annotated heatmap (`Hour` × `channel group`, `YlGnBu` palette)
- **Insight:** Reveals peak traffic windows per channel to inform optimal publishing and campaign scheduling

#### Q7 — Engagement Rate vs Sessions Over Time
- **Chart:** Dual line plot overlaying `Engagement rate` and `Sessions`
- **Insight:** Determines whether traffic volume positively, negatively, or independently correlates with quality of engagement

---

## How to Run

1. Place `data-export (1) (1).csv` in `/content/` (Google Colab) or update the file path in the notebook.
2. Open `EDA_website_Project2.ipynb` in **Google Colab** or **Jupyter Notebook**.
3. Run all cells sequentially (`Runtime → Run all` in Colab).

### Requirements

```bash
pip install numpy pandas matplotlib seaborn
```

> **Python version:** 3.8 or above recommended.

---

## Key Insights

- **Organic Search** typically dominates user acquisition, making SEO a priority investment.
- Channels with **high engagement time** (e.g., Direct, Email) indicate loyal, intent-driven audiences.
- **Engagement rate distributions** vary significantly — some channels show high variability, suggesting inconsistent content or audience fit.
- The **hourly heatmap** exposes clear peak windows (often mid-morning and early evening) that differ by channel, enabling smarter scheduling.
- The **sessions vs. engagement rate** trend reveals whether traffic surges are quality-driven or volume-driven — a critical distinction for campaign optimisation.

---

## Future Enhancements

- Integrate **conversion rate** data to connect engagement metrics with business outcomes.
- Apply **time-series forecasting** (e.g., Prophet, ARIMA) to predict future traffic trends.
- Build an **interactive dashboard** using Plotly or Streamlit for stakeholder reporting.
- Extend analysis to include **device category** (mobile vs. desktop) breakdown.

---

## Author

Professional EDA project on Website Performance Analytics using Python, Pandas, and Seaborn.
