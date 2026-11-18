# SupeRR Selector Season 2: Pressure Performance Intelligence & IPL Auction Scouting System

> **Rajasthan Royals Decision-Support & Auction Intelligence Platform**  
> *Target: Identifying clutch batters and bowlers for the IPL 2027 Mega-Auction*  
> *Dataset Universe: 2022–2026 (5 IPL Seasons + 134 T20I Series | 193,988 Processed Deliveries)*

---

## 📌 Executive Overview

In T20 cricket, champions are crowned not by aggregate statistics compiled in comfortable phases, but by **who executes when the match is on the line**. 

Conventional franchise scouting often falls into the trap of over-indexing on nominal runs or raw wickets. The **SupeRR Selector** system introduces the **Pressure Intelligence Index (PII)** — an explainable, statistically sound framework that:
1. **Isolates true high-pressure delivery contexts** dynamically using required run rate, balls remaining, wickets in hand, and closeness — *with zero post-match information leakage*.
2. **Measures the Pressure Delta** ($\Delta_{\text{Perf}} = \text{Pressure} - \text{Normal}$) to identify players who actively elevate their game.
3. **Applies James-Stein Empirical Bayesian Shrinkage** ($k_{\text{bat}} = 100$, $k_{\text{bowl}} = 60$) to eliminate small-sample bias and reward proven, repeatable performance.
4. **Evaluates match-to-match consistency** using the Coefficient of Variation ($CV$) and classifies players into four strategic performance quadrants (*Elite, Reliable, Streaky, Inconsistent*).
5. **Generates actionable scouting cards** tailored to Rajasthan Royals' squad balance and tactical requirements for the IPL 2027 auction.

---

## 🏆 Official Top 10 Rankings

### 🏏 Top 10 Batters Under Pressure (Batting PII)
| Rank | Player Name | Batting PII | HP Balls | HP Strike Rate | Normal SR | SR Delta | Reliability | Quadrant |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Rinku Singh** | **91.95** | 197 | 196.4 | 119.5 | **+76.9** | 0.66 | Elite |
| **2** | **Tim David** | **88.80** | 191 | 207.3 | 173.0 | **+34.4** | 0.66 | Elite |
| **3** | **Ayush Badoni** | **88.19** | 117 | 196.6 | 122.1 | **+74.5** | 0.54 | Elite |
| **4** | **Tristan Stubbs** | **84.51** | 206 | 194.2 | 121.3 | **+72.9** | 0.67 | Elite |
| **5** | **Tilak Varma** | **81.77** | 198 | 193.4 | 123.6 | **+69.8** | 0.66 | Elite |
| **6** | **David Miller** | **78.26** | 154 | 194.8 | 117.4 | **+77.4** | 0.61 | Elite |
| **7** | **Nicholas Pooran** | **77.81** | 121 | 201.7 | 153.5 | **+48.2** | 0.55 | Elite |
| **8** | **Rovman Powell** | **75.99** | 241 | 187.1 | 133.9 | **+53.2** | 0.71 | Elite |
| **9** | **MS Dhoni** | **75.89** | 216 | 168.1 | 81.8 | **+86.2** | 0.68 | Elite |
| **10** | **Jason Holder** | **74.59** | 146 | 183.6 | 72.2 | **+111.3** | 0.59 | Elite |

### 🎯 Top 10 Bowlers Under Pressure (Bowling PII)
| Rank | Player Name | Bowling PII | HP Balls | HP Economy | Normal Econ | Econ Delta | Reliability | Quadrant |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **Haris Rauf** | **97.91** | 96 | **7.56** | 9.25 | **+1.69** | 0.62 | Elite |
| **2** | **Wanindu Hasaranga** | **78.26** | 111 | **7.57** | 7.23 | -0.34 | 0.65 | Elite |
| **3** | **Eshan Malinga** | **75.06** | 119 | 9.03 | 9.30 | **+0.28** | 0.66 | Elite |
| **4** | **Abbas Afridi** | **70.39** | 63 | 10.00 | 10.31 | **+0.31** | 0.51 | Reliable |
| **5** | **Prince Yadav** | **69.03** | 101 | 8.44 | 8.71 | **+0.28** | 0.63 | Elite |
| **6** | **Ravi Bishnoi** | **68.38** | 74 | 9.32 | 7.57 | -1.75 | 0.55 | Reliable |
| **7** | **Mohammad Rishad Hossain** | **68.27** | 63 | 9.24 | 6.49 | -2.75 | 0.51 | Reliable |
| **8** | **Noor Ahmad** | **68.16** | 76 | 8.37 | 7.38 | -0.99 | 0.56 | Elite |
| **9** | **Harshal Patel** | **67.64** | 203 | 9.81 | 8.83 | -0.98 | 0.77 | Elite |
| **10** | **Vaibhav Arora** | **67.19** | 88 | 9.95 | 8.47 | -1.49 | 0.59 | Reliable |

---

## 📂 Project Architecture

```
SupeRR_Selector/
├── data/
│   ├── raw/                           # Raw input CSV datasets
│   └── processed/                     # Cleaned, enriched & engineered parquet/csv tables
├── notebooks/                         # 7 Explanatory Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_pressure_definition.ipynb
│   ├── 04_batting_analysis.ipynb
│   ├── 05_bowling_analysis.ipynb
│   ├── 06_reliability_analysis.ipynb
│   └── 07_player_rankings.ipynb
├── src/                               # Modular analytics codebase
│   ├── data_loader.py                 # Robust data loading & schema validation
│   ├── preprocessing.py               # Cricket accounting & delivery engineering
│   ├── pressure_engine.py             # Pressure Intelligence Engine (PII)
│   ├── batting_model.py               # Batting metrics & pressure delta
│   ├── bowling_model.py               # Bowling metrics, phase & dot efficiency
│   ├── reliability_model.py           # James-Stein shrinkage & bootstrap CIs
│   ├── consistency_model.py           # CV calculation & 4-quadrant mapping
│   ├── ranking_engine.py              # PII scoring synthesis & sensitivity tests
│   └── scouting_engine.py             # RR scouting dossiers & tactical fit
├── dashboard/                         # Interactive Streamlit Scouting Platform
│   ├── App.py                         # Main landing application
│   ├── theme.py                       # Rajasthan Royals visual theme & helpers
│   └── pages/
│       ├── 1_Executive_Overview.py
│       ├── 2_Pressure_Intelligence.py
│       ├── 3_Batting_Analytics.py
│       ├── 4_Bowling_Analytics.py
│       ├── 5_Player_Explorer.py
│       ├── 6_Reliability_and_Consistency.py
│       └── 7_RR_Scouting_Recommendations.py
├── reports/                           # Output deliverables
│   ├── methodology_report.pdf         # Professional PDF methodology report
│   ├── methodology_report.md          # Full markdown methodology report
│   ├── player_rankings.csv            # Official export of all ranked players
│   └── scouting_recommendations.csv   # Structured RR auction shortlist
├── tests/                             # Comprehensive test suite (55 unit tests)
│   ├── test_preprocessing.py          # Cricket rule validation
│   ├── test_pressure_engine.py        # Leakage-free pressure testing
│   ├── test_batting_metrics.py        # Batting formulas & sample gates
│   ├── test_bowling_metrics.py        # Bowling formulas & phase allocation
│   └── test_ranking_engine.py         # Bounds, uniqueness & monotonicity
├── run_pipeline.py                    # Master pipeline orchestrator script
├── requirements.txt                   # Dependency specification
└── README.md                          # Platform documentation
```

---

## 🚀 Installation & Execution

### 1. Prerequisites
- Python 3.9+
- Recommended: clean virtual environment

```bash
git clone <repo-url>
cd SupeRR_Selector
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Analytics Pipeline
To execute data ingestion, cleaning, pressure calculation, shrinkage, and ranking:
```bash
python run_pipeline.py
```
*Expected runtime: ~15 to 25 seconds on modern multi-core machines.*

### 3. Run Automated Unit Tests
To verify all 55 test assertions:
```bash
python3 -m pytest tests/ -v
```

### 4. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/App.py
```
Open `http://localhost:8501` in your browser.

---

## 🔬 Core Methodology Highlights

### 1. Dynamic Match-State Pressure Formulation
Pressure is calculated on a per-delivery basis using **strictly prior match state**:
- **Batting (Chasing)**:
  $$\text{Score}_{\text{bat}} = 0.35 \cdot \text{Norm}(eRRR) + 0.20 \cdot \text{Norm}(\text{Runs Needed}) + 0.20 \cdot \text{Norm}(\text{Balls Rem.}^{-1}) + 0.15 \cdot \text{Norm}(\text{Wickets Lost}) + 0.10 \cdot \text{Closeness}$$
- **Bowling (Defending)**:
  $$\text{Score}_{\text{bowl}} = 0.35 \cdot \text{Norm}(\text{Defend Rate}) + 0.20 \cdot \text{Norm}(\text{Balls Rem.}^{-1}) + 0.20 \cdot \text{Norm}(\text{Opp. Wickets in Hand}) + 0.15 \cdot \text{Momentum} + 0.10 \cdot \text{Closeness}$$

### 2. Cricket Accounting Integrity
- **Wides & No-balls** are properly classified as extra deliveries, not legal balls.
- **Byes & Leg-byes** are credited to team totals and omitted from bowler runs.
- **Run-outs & Retire-outs** are never credited to bowlers.
- **Super overs & Emerging team cups** are cleanly segmented to prevent dilution.

### 3. Empirical Bayesian Shrinkage
$$\text{Adjusted Metric} = \left(\frac{n}{n + k}\right) \cdot \text{Observed} + \left(\frac{k}{n + k}\right) \cdot \text{Baseline}$$
- Batting constant $k = 100$ balls faced.
- Bowling constant $k = 60$ legal balls bowled.
- Players with reliability weight $< 0.40$ receive an evidence-strength penalty.

---

## 🎯 Rajasthan Royals Tactical Recommendations (IPL 2027)

1. **Target Rinku Singh or Tim David as Priority Finisher**:
   - RR's historical bottleneck has been scoring velocity in high-pressure chases (overs 17–20).
   - Rinku Singh's **+76.9 SR Delta** and Tim David's **207.3 HP Strike Rate** represent transformative chasing firepower.
2. **Target Haris Rauf for High-Pace Death Execution**:
   - Restricts batting attacks to **7.56 rpo** under pressure (+1.69 rpo improvement over normal phases).
   - Pairs with Sandeep Sharma to form an impenetrable death-bowling pair.
3. **Target Wanindu Hasaranga for Clutch Middle-Overs Spin**:
   - Exceptional **7.57 rpo** pressure economy chokes boundary momentum and creates wicket-taking opportunities at Sawai Mansingh Stadium.

---

## 📄 Deliverable Files
- `reports/methodology_report.pdf`: Publication-ready PDF report with tables and analytical explanations.
- `reports/methodology_report.md`: Complete Markdown methodology documentation.
- `reports/player_rankings.csv`: Complete rankings file.
- `reports/scouting_recommendations.csv`: Actionable RR scouting dossiers.
