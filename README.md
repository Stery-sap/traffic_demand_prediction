# 🚦 Traffic Demand Prediction

> Predicting urban traffic demand using machine learning — a competition solution achieving **~87.3 R² score** (out of 100).

---

## 📌 Problem Statement

Cities worldwide are increasingly turning to AI-powered solutions to tackle traffic congestion. This project builds a regression model to predict **traffic demand** at various geographic locations given contextual information such as time of day, road characteristics, and environmental conditions.

**Evaluation Metric:**
```
score = max(0, 100 * r2_score(actual, predicted))
```

---

## 📁 Repository Structure

```
├── traffic_demand_prediction.ipynb   # Main ML pipeline 
├── train.csv                         # Training data (77,299 × 11)
├── test.csv                          # Test data (41,778 × 10)
├── submission.csv                    # Final predictions (generated)
└── README.md
```

---

## 📊 Dataset

| File | Rows | Columns | Notes |
|------|------|---------|-------|
| `train.csv` | 77,299 | 11 | Includes `demand` target |
| `test.csv` | 41,778 | 10 | Demand to be predicted |

### Column Descriptions

| Column | Description |
|--------|-------------|
| `Index` | Unique row identifier |
| `geohash` | Geographic location encoding |
| `day` | Day when the record was captured |
| `timestamp` | Time of record in `H:MM` format |
| `RoadType` | Type of road (Residential / Street / Highway) |
| `NumberofLanes` | Number of lanes at the location |
| `LargeVehicles` | Whether large vehicles are permitted |
| `Landmarks` | Whether landmarks are nearby |
| `Temperature` | Temperature at the location |
| `Weather` | Weather condition (Sunny / Rainy / Cloudy / Foggy / Snowy) |
| `demand` | *(Target)* Traffic demand at the timestamp |

---

## ⚙️ Feature Engineering

20 features were engineered from the 10 raw input columns across four categories:

### 🕐 Temporal (7 features)
| Feature | Description |
|---------|-------------|
| `hour`, `minute`, `time_of_day` | Parsed from `H:MM` timestamp string |
| `hour_sin`, `hour_cos` | Cyclical encoding — treats hour 23 → 0 as continuous |
| `day_sin`, `day_cos` | Cyclical weekly-day encoding |
| `is_rush_hour` | 1 if 7–9 AM or 5–7 PM, else 0 |
| `is_night` | 1 if hour ≥ 22 or ≤ 5, else 0 |

### 📍 Geospatial — Target Encoding (4 features)
| Feature | Description |
|---------|-------------|
| `geo_demand_mean` | Mean demand per geohash (from training data) |
| `geo_demand_std` | Std dev of demand per geohash |
| `prefix_demand_mean` | Mean demand for the first 4 chars of geohash (coarser cluster) |
| `geohash_len` | Length of geohash string (proxy for location precision) |

> ⚠️ Target encoding uses only training labels. Unseen geohashes in test fall back to the global mean to prevent data leakage.

### 🛣️ Road (4 features)
| Feature | Description |
|---------|-------------|
| `NumberofLanes` | Used as-is |
| `RoadType_enc` | Ordinal: Unknown=0, Residential=1, Street=2, Highway=3 |
| `LargeVehicles_enc` | Binary: 1 if Allowed |
| `Landmarks_enc` | Binary: 1 if Yes |

### 🌤️ Environmental (2 features)
| Feature | Description |
|---------|-------------|
| `Temperature` | Numeric; NaN filled with column median |
| `Weather_enc` | Ordinal: Unknown=0, Sunny=1, Cloudy=2, Rainy=3, Foggy=4, Snowy=5 |

---

## 🤖 Models & Results

Four models were evaluated using **5-fold cross-validation** with R² as the scoring metric:

| Model | CV R² (Mean) | Std Dev | Score (0–100) |
|-------|:-----------:|:-------:|:-------------:|
| Ridge Regression | 0.7788 | ±0.0954 | 77.9 |
| Random Forest | 0.8732 | ±0.0740 | 87.3 |
| XGBoost | 0.8697 | ±0.0826 | 87.0 |
| **LightGBM ✅** | **0.8733** | **±0.0864** | **87.3** |

**LightGBM** was selected as the final model — highest mean R², fast training, and excellent handling of structured tabular data.

### Why the big jump from Ridge → Ensembles?
The ~9-point gap confirms that demand patterns are strongly **nonlinear** — particularly the interactions between geohash location, hour of day, and road type that a linear model cannot capture.

---

## 🚀 How to Run


### Clone the repository

```bash
git clone https://github.com/your-username/traffic-demand-prediction.git
cd traffic-demand-prediction
```

### Install dependencies
```bash
pip install pandas numpy scikit-learn lightgbm xgboost
```

##  Running on Google Colab

```markdown
1. Download `traffic_demand_prediction.ipynb`.
2. Open Google Colab.
3. Upload the notebook.
4. Upload `train.csv` and `test.csv` to the Colab session or mount Google Drive.
5. Update dataset paths if necessary.
6. Run all cells.
```

### Output
The script will:
- Print 5-fold CV R² scores for all 4 models
- Display the best model
- Save `submission.csv` ready for competition upload

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, manipulation, target encoding |
| `numpy` | Cyclical encoding, array operations |
| `scikit-learn` | Ridge, RandomForest, cross-validation, R² scoring |
| `xgboost` | XGBRegressor |
| `lightgbm` | LGBMRegressor — final model |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
