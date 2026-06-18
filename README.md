# Airline Customer Satisfaction Prediction Engine
## Binomial Logistic Regression Inference & Preprocessing Pipeline

### 📋 Project Overview
This project establishes an end-to-end machine learning and data science pipeline designed to classify and predict airline passenger satisfaction. Using passenger survey data, flight distances, and operational latency trackers, we optimize a binomial logistic regression model to isolate key elements driving customer experience and detect systemic service friction points.

---

### 📊 Performance Summary Matrix
By training our optimized binomial classification engine over the complete flight dataset, the predictive model achieves the following classification thresholds on the validation holdout split:

* **Global Classification Accuracy:** `82.53%` (High overall baseline precision)
* **Precision Profile:** `83.93%` (Minimizes false-positive satisfaction alerts)
* **Sensitivity / Recall Rate:** `84.20%` (Reliably flags operational points of customer friction)
* **Harmonic Mean F1-Score:** `84.07%` (Confirms a high, well-balanced classification stability between precision and recall thresholds)

---

### 📉 Core Visualizations & Model Verification

#### 1. Performance Confusion Matrix
The confusion matrix below maps our test predictions directly against the true recorded customer feedback categories:



* **True Negatives (Actual Dissatisfied, Predicted Dissatisfied):** `9,436` passengers accurately flagged as churn or friction risks.
* **False Positives (Actual Dissatisfied, Predicted Satisfied):** `2,285` records misclassified, yielding a controlled type-I error surface.
* **True Positives (Actual Satisfied, Predicted Satisfied):** `11,937` passengers correctly matched to positive service tiers.

#### 2. Sigmoid Variable Curve Validation
To satisfy the fundamental mathematical assumptions of binomial logistic regression, our primary continuous numerical predictor (`Inflight entertainment`) was mapped against the computed output probabilities:



This visual confirmation highlights the non-linear S-curve transition boundary across variance tiers, justifying our choice of a binomial modeling framework over a standard linear setup.

---

### 🛠️ Data Engineering & Pipeline Transformations

To ensure absolute compliance with predictive modeling criteria, the data preparation lifecycle was executed using a repeatable `scikit-learn` engineering pipeline:

1.  **Explicit Missing Data Strategy:** The initial baseline audit identified `393` incomplete or null entries exclusively within the `Arrival Delay in Minutes` feature column. To prevent introducing mathematical imputation bias across high-variance timeline metrics, these rows were completely removed from the frame. This step stabilized our continuous feature profiles across a finalized dataset of `129,487` operational profiles.
2.  **Target Categorical Matrix Formulation:** Passenger response text variables are mapped directly into standardized binary logical structures:
    $$\text{satisfied} \longrightarrow 1 \quad | \quad \text{dissatisfied} \longrightarrow 0$$
3.  **Categorical Feature Dummy Transformations:** Multi-class strings (`Customer Type`, `Type of Travel`, `Class`) are mapped through an automated One-Hot Encoding process using a `drop='first'` parameter setting. This structural constraint eliminates perfect multicollinearity (the dummy variable trap) and ensures stable coefficient weights.
4.  **Standard Variance Scaling:** All continuous numerical features and scalar review scores are transformed via Z-score scaling. This standardizes variance scales and keeps large, high-magnitude metrics (like `Flight Distance`) from overpowering smaller numerical fields:
    $$z = \frac{x - \mu}{\sigma}$$
5.  **Validation Configuration:** The preprocessed matrix is split using a strict, stratified **80% training / 20% validation split** (`random_state=42`) to verify the generalization performance of our coefficients.

---

### 🧬 Parameter Significance Coefficients & Strategic Insights

* **Model Structural Intercept ($\beta_0$):** `1.3104`

| Rank | Model Feature Variable | Log-Odds Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Strategic Impact Profile |
| :--- | :--- | :---: | :---: | :--- |
| 1 | `Inflight entertainment` | `0.9736` | `2.6475` | **Primary Driver:** A 1-unit increase multiplies satisfaction odds by **2.65x** ($+164.75\%$). |
| 2 | `Seat comfort` | `0.4088` | `1.5049` | High impact physical comfort indicator ($+50.49\%$ odds lift per unit). |
| 3 | `On-board service` | `0.3991` | `1.4904` | Critical point of service delivery interaction. |
| 4 | `Inflight wifi service` | `-0.1202` | `0.8868` | Shows a minor negative correlation when isolated alongside entertainment features. |
| 5 | `Arrival Delay in Minutes` | `-0.3042` | `0.7377` | **Operational Friction Point:** Delays create a measurable drag on customer scores. |
| 6 | `Class_Eco` | `-0.7256` | `0.4840` | Economy tier passengers exhibit a $51.60\%$ drop in satisfaction odds vs Business. |
| 7 | `Type of Travel_Personal Travel`| `-0.7581` | `0.4686` | Personal/Leisure passengers are significantly tougher to satisfy than business travelers. |
| 8 | `Customer Type_disloyal Customer`| `-1.8906` | `0.1510` | Non-retained, casual consumers represent our single highest structural attrition group. |
---

### 🔮 Data-Backed Business Recommendations

Based on the validated parameters optimized from the flight dataset, our operational guidance for executive leadership is structured as follows:

1.  **Prioritize High Odds-Ratio Asset Capitalization:** Strategic resource investments should prioritize digital infrastructure—specifically **`Inflight entertainment` systems**—over basic cabin changes. Because a single-unit rating improvement on entertainment provides the single largest mathematical lift to passenger satisfaction odds ($e^\beta = 2.65$), it represents our highest-leverage area for deployment.
2.  **Mitigate Latency Bottlenecks with Proactive Automation:** **`Arrival Delay in Minutes`** ($\beta = -0.3042$) acts as a measurable service drag. Flight control should link tracking dashboards to automated customer retention engines. The moment an arrival delay crosses a critical threshold, the system can automatically push loyalty miles or lounge passes to affected passengers to neutralize the negative satisfaction drop before arrival.
3.  **Segment-Specific Retention Strategies:** The negative offsets generated across dummy baselines—specifically **`Customer Type_disloyal Customer`** ($\beta = -1.8906$) and **`Class_Eco`** ($\beta = -0.7256$)—confirm a steep satisfaction gap between economy/casual flyers and loyal business class accounts. This empirical drop justifies targeted service changes, such as providing economy segments with complimentary entertainment access or early booking options to help close the customer satisfaction gap.
