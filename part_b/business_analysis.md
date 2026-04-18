# Business Case Analysis - Promotion Effectiveness at a Fashion Retail Chain

## B1. Problem Formulation

### B1(a) - ML Problem Formulation

**Target Variable:** items_sold - number of items sold per store per month.

**Input Features:**

| Feature | Type |
|---|---|
| promotion_type | Categorical |
| store_size, location_type | Categorical |
| is_weekend, is_festival, is_month_end | Binary |
| competition_density, month, day_of_week | Numerical |

**Problem Type:** Supervised Regression.

**Justification:** items_sold is a continuous numerical value and we have
labeled historical data. The goal is to predict a quantity - not a class -
making this a regression problem. The model can then recommend whichever
promotion maximises predicted sales for each store.

### B1(b)  Why items_sold Over Revenue

Revenue = Items Sold x Price. Since promotions like Flat Discount and BOGO
directly reduce price, revenue can fall even when more items are sold.
This makes revenue an unreliable signal for promotion effectiveness.

items_sold measures customer response directly, independent of price
changes  giving a consistent, comparable signal across all stores and
promotion types.

**Broader Principle:** The target variable must directly measure what the
business wants to optimise - not a proxy distorted by confounding factors.
This is known as **metric misalignment**, a common pitfall in real-world ML.

### B1(c)  Alternative to a Single Global Model

A global model assumes all 50 stores respond to promotions identically 
which is incorrect. Urban stores have brand-conscious customers while
rural stores are price-sensitive.

**Proposed Strategy - Location-Stratified Models:**

| Model | Trained On |
|---|---|
| Urban Model | All urban stores |
| Semi-Urban Model | All semi-urban stores |
| Rural Model | All rural stores |

Each sub-model learns promotion response patterns specific to that store
context, producing more accurate and actionable predictions. This follows
the **stratification principle** - when subgroups behave fundamentally
differently, model them separately.

## B2. Data and EDA Strategy

### B2(a)  Joining the Four Tables

**Join Strategy:**

| Join | Key | Type |
|---|---|---|
| transactions + store attributes | store_id | Left Join |
| result + promotion details | promotion_type | Left Join |
| result + calendar | transaction_date | Left Join |

**Grain of Final Dataset:**
One row = one store x one month x one promotion type.
Each row represents total items sold at a specific store in a specific
month under a specific promotion.

**Aggregations before modelling:**
1. items_sold - sum per store per month
2. competition_density- taken as-is (store-level attribute)
3. is_weekend`, `is_festival - count of such days in that month

### B2(b) - EDA Strategy

1. Distribution of items_sold (Histogram)
Check for skewness or outliers. If heavily skewed, apply log transformation
before modelling.

2. Items Sold by Promotion Type (Box Plot)
Compare median sales across all 5 promotion types. Identifies which
promotions are strongest - informs feature importance expectations.

3. Items Sold by Location Type (Bar Chart)
Check if urban/semi-urban/rural stores respond differently to promotions.
If yes - justifies building stratified models instead of one global model.

4. Correlation Heatmap (Numerical Features)
Check multicollinearity between features like month, is_festival,
is_weekend. Highly correlated features may need to be dropped or combined.

5. Items Sold over Time (Line Chart)
Detect seasonal trends - peaks during festival months or year-end.
Informs whether month and is_festival are strong features.

### B2(c) - Promotion Imbalance (80% No-Promotion)

Problem:
If 80% of rows have no promotion, the model sees very few promoted
examples. It may learn to predict average sales regardless of promotion
type - making it useless for promotion recommendation.

Steps to Address:

1. **Oversample promoted transactions** using SMOTE or random oversampling
   so the model sees more promotion examples during training.

2. **Filter and model separately** - build one model exclusively on
   promoted transactions to learn promotion-specific patterns clearly.

3. **Add a binary feature `has_promotion`** (0/1) so the model explicitly
   learns the baseline difference between promoted and non-promoted sales.

4. **Stratified sampling** during train/test split to ensure both sets
   contain a representative proportion of promoted transactions.

## B3. Model Evaluation and Deployment

### B3(a) - Train-Test Split and Evaluation Metrics

**How to Split:**
Data spans 3 years (36 months) across 50 stores.
Use a **temporal split** - train on older data, test on most recent data.

| Set | Period | Records |
|---|---|---|
| Training | Month 1 - Month 30 (83%) | ~2,500 rows |
| Test | Month 31 - Month 36 (17%) | ~300 rows |

This ensures the model is always predicting future months from past data -
mirroring real deployment conditions.

Why Random Split is Inappropriate:
Monthly retail data is time-ordered. A random split allows future
transactions to appear in the training set, giving the model unfair
knowledge of future patterns. This is **data leakage** - the model
appears accurate in evaluation but fails in production.

Evaluation Metrics:

| Metric | Formula | Business Interpretation |
|---|---|---|
| **MAE** | Mean Absolute Error | On average, prediction is off by X items - easy to explain to marketing team |
| **RMSE** | Root Mean Squared Error | Penalises large errors - useful if a bad recommendation causes significant overstock |
| **R²** | Variance explained | How much of sales variation the model captures - higher is better |

For this business problem, **MAE is most useful** - the marketing team
can directly understand "our recommendations are off by ~15 items on average."

### B3(b) - Investigating Different Recommendations for Same Store

The model recommends **Loyalty Points in December** and **Flat Discount in March**
for Store 12. This is explained by the top features driving each prediction.

Investigation Steps:

1. Extract feature importance from the Random Forest model to identify
   which features matter most globally (e.g., is_festival, month).

2. Use **SHAP values** for Store 12 specifically - SHAP shows how much
   each feature pushed the prediction up or down for that exact row.

3. Compare the two months:

| Feature | December | March | Effect |
|---|---|---|---|
| is_festival | 1 (festival season) | 0 | Boosts Loyalty Points in Dec |
| month| 12 (peak season) | 3 (off-season) | Higher baseline in Dec |
| competition_density | Low | High | Flat Discount more competitive in Mar |
| is_weekend | More weekends | Fewer weekends | More footfall in Dec |

**Communication to Marketing Team:**
 "In December, festival season and high footfall make customers more likely to return - so Loyalty Points drives repeat visits effectively.
 In March, footfall is lower and competition is higher, so a Flat Discount attracts price-sensitive customers more reliably."

This converts model logic into a simple, actionable business narrative.

### B3(c) - End-to-End Deployment Process

**Step 1 - Save the Model**
After training, serialize the full pipeline using joblib:
python
import joblib
joblib.dump(pipeline, 'promotion_model.pkl')

The entire pipeline (preprocessor + model) is saved together -
ensuring new data is transformed identically to training data.

**Step 2 - Monthly Data Preparation**
At the start of each month, prepare one row per store with:
1.Current store_size, location_type, competition_density
2.Calendar features: month, is_festival, is_weekend count
3.One row per promotion type (5 rows per store x 50 stores = 250 rows)

Load and predict:
```python
model = joblib.load('promotion_model.pkl')
predictions = model.predict(new_month_data)
# Recommend promotion with highest predicted items_sold per store
```
**Step 3 - Generate Recommendations**
For each store, select the promotion type with the highest predicted
items_sold- output as a simple table for the marketing team.

**Step 4 - Monitoring for Model Degradation**

| Monitor | Method | Threshold |
|---|---|---|
| **Prediction Drift** | Compare predicted vs actual items_sold each month | MAE increases >20% from baseline |
| **Data Drift** | Check if input feature distributions shift | New store sizes or promotion types appear |
| **Business Metric** | Track actual sales after recommended promotion | If sales drop consistently, trigger review |

**When to Retrain:**
1. MAE on recent months exceeds acceptable threshold
2. New promotion types are introduced
3. Major external change occurs (new competitor, economic shift)
4. Every 6 months as a scheduled retraining cycle