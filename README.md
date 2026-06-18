# Airline Customer Satisfaction Prediction Pipeline
## Binomial Logistic Regression Inference Framework

### 📊 Executive Summary Performance Overview
By shifting from basic tracking statistics to an active, probability-driven predictive framework, the modeling pipeline achieves strong statistical sorting with verified, high-level structural patterns:

* **Global Classification Accuracy:** `96.50%`
* **Precision Profile:** `95.33%` (minimizes false satisfaction warnings)
* **Sensitivity / Recall Rate:** `100.00%` (captures passenger friction trends completely)

---

### 📉 Model Visualizations & Matrix Analysis

#### 1. Confusion Matrix
The confusion matrix below illustrates the accurate distribution of our predictions compared to actual passenger data on the evaluation test set:


* **True Negatives (Actual Dissatisfied, Predicted Dissatisfied):** 50 passengers accurately flagged as friction risks.
* **False Positives (Actual Dissatisfied, Predicted Satisfied):** 7 passengers, establishing a solid balance with minimal type-1 errors.
* **True Positives (Actual Satisfied, Predicted Satisfied):** 143 passengers correctly classified.

#### 2. Log-Odds Feature Weights Analysis
The variable behavior map below visualizes which touchpoints have the largest impact on whether a passenger has a positive or negative experience:


---

### ⚙️ Architecture and Feature Transformation Pipeline

1. **Target Feature Label Conversion:** Maps the qualitative evaluation arrays directly into explicit numerical indices:
   $$\text{Satisfied} \longrightarrow 1 \quad | \quad \text{Dissatisfied} \longrightarrow 0$$

2. **Categorical Feature Dummy Encoding:** Categorical features (`Customer_Type`, `Class`) are split into independent variables using structural dummy configurations (`drop='first'`). This prevents multicollinearity and ensures stable data variance processing during training.

3. **Standard Variance Scaling:** Continuous operational tracking vectors (`Age`, `Flight_Distance`, `Delays`) are normalized to a standard scale using a `StandardScaler` matrix optimization tool:
   $$z = \frac{x - \mu}{\sigma}$$
   This step prevents large raw numbers (such as long flight mileages) from overwhelming smaller but critically important survey metrics during model tuning.

---

### 🧬 Strategic Observations and Odds Ratio Drivers

When we convert log-odds values back into exponential scale matrices ($OR = e^\beta$), we can extract clear, data-driven business insights:

* **Inflight Digital Infrastructure:** `Inflight_Wifi_Quality` stands out as a primary driver of positive reviews. Improving this service by a single rating unit increases passenger satisfaction odds significantly.
* **Premium Cabin Comfort:** Choosing a **Business Class** ticket shows a strong positive correlation with satisfaction, highlighting the value of premium tier features.
* **Operational Friction Points:** `On_time_Departure_Delay` shows a clear negative weight, indicating that flight delays directly drop customer satisfaction scores.
