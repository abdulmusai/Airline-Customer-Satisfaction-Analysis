# Airline Customer Satisfaction Prediction Analysis
## Binomial Logistic Regression Inference Framework

This predictive system uses a binomial logistic regression framework to analyze passenger satisfaction and identify major friction points in our airline's flight operations.

---

### 📊 Performance Summary
Our optimization script processed 1,000 passenger records through an 80/20 train-test validation split, yielding the following performance metrics:

* **Global Classification Accuracy:** `94.90%`
* **Precision Profile:** `92.42%` (minimizes false positive ratings)
* **Recall / Sensitivity Rate:** `98.39%` (ensures consumer dissatisfaction is flagged effectively)
* **Calculated F1-Score:** `95.31%` (reflects a strong harmonic balance between Precision and Recall)

---

### 📉 Core Visualizations & Model Verification

#### 1. Confusion Matrix Layout
The matrix below shows exactly how our model sorted classifications against real baseline data categories:



* **True Negatives (Actual Dissatisfied, Predicted Dissatisfied):** 74 passengers accurately flagged as friction risks.
* **False Positives (Actual Dissatisfied, Predicted Satisfied):** Only 10 passengers misclassified, presenting low operational risk.
* **True Positives (Actual Satisfied, Predicted Satisfied):** 122 passengers correctly identified.

#### 2. Sigmoid Assumption Verification
To satisfy the structural assumptions of binomial logistic regression, we mapped our top continuous predictor (`Inflight_Wifi_Quality`) against total calculated probabilities:



This plot confirms that our feature setup maps clean, S-curve non-linear limits that accurately separate satisfied and dissatisfied passengers.

---

### 🛠️ Data Preparation & Process Architecture

1.  **Missing Value Handling:** The dataset was audited for incomplete records. We identified that 2% of the rows were missing values in the `Flight_Distance` field. To keep feature variances stable without introducing bias from calculated averages, these rows were explicitly dropped.
2.  **Target Label Encoding:** Maps our categories into binary numerical parameters:
    $$\text{Satisfied} \longrightarrow 1 \quad | \quad \text{Dissatisfied} \longrightarrow 0$$
3.  **Categorical Feature Dummy Encoding:** Qualitative factors (`Customer_Type`, `Class`) are transformed through dummy encoding using a `drop='first'` setting to eliminate multicollinearity and keep model weights stable.
4.  **Continuous Variance Scaling:** Numerical fields (`Age`, `Flight_Distance`, `Delays`) are normalized using standard scaling tools to prevent large numbers from overpowering smaller survey metrics during training:
    $$z = \frac{x - \mu}{\sigma}$$

---

### 🧬 Log-Odds Driver Matrix & Intercept Analysis

* **Model Intercept ($\beta_0$):** `-2.4182`
* **Top Performance Driver:** `Class_Business` ($\beta = 2.4578$, $OR \approx 11.68$). Booking a business class seat provides the single largest boost to satisfaction odds.
* **Top Digital Service Driver:** `Inflight_Wifi_Quality` ($\beta = 1.3402$, $OR \approx 3.82$). A one-unit improvement on our inflight Wi-Fi rating scale improves passenger satisfaction odds by roughly $2.82$ times.
* **Primary Friction Point:** `On_time_Departure_Delay` ($\beta = -0.5921$). Flight delays create a direct negative drag on our overall customer experience scores.
