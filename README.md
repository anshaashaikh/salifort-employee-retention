# Salifort Motors Employee Retention Analysis

A capstone data science project analyzing employee turnover at Salifort Motors
and building predictive models to identify employees at risk of leaving. This
project represents the culmination of the Google Advanced Data Analytics
Certificate (Course 7).

---

## Business Problem

The HR department at Salifort Motors collected employee data but lacked the
analytical capacity to act on it. Leadership needed to understand: **what
factors make an employee likely to leave, and can we predict it?**

Predicting attrition before it happens enables proactive retention — saving
the significant time and cost of recruiting, interviewing, and onboarding
replacements.

---

## Dataset

**Source:** HR capstone dataset (fictional, pedagogical)
**Size:** 14,999 rows × 10 columns (3,008 duplicates removed → 11,991 clean rows)

| Variable | Description |
|---|---|
| `satisfaction_level` | Self-reported job satisfaction [0–1] |
| `last_evaluation` | Score from most recent performance review [0–1] |
| `number_project` | Number of projects contributed to |
| `average_monthly_hours` | Average hours worked per month |
| `tenure` | Years at the company |
| `work_accident` | Whether employee had a workplace accident |
| `left` | Target variable: whether employee left (1) or stayed (0) |
| `promotion_last_5years` | Whether promoted in the last 5 years |
| `department` | Employee's department |
| `salary` | Salary level (low / medium / high) |

**Class balance:** 83.4% stayed, 16.6% left — moderately imbalanced, no
resampling required.

---

## Key EDA Findings

- **All employees with 7 projects left the company.** Employees handling
  3–4 projects had the lowest churn rate.
- **Three distinct churn profiles emerged** from the hours vs. satisfaction
  scatterplot: overworked and burned out (very low satisfaction, 240–315
  hrs/month), moderately dissatisfied (normal hours, ~0.4 satisfaction),
  and high performers who may have been poached (high hours, high
  satisfaction).
- **Four-year employees show unusually low satisfaction**, suggesting a
  specific policy or culture issue at that tenure milestone.
- **Employees working long hours are rarely promoted.** Almost no employee
  who worked the most hours received a promotion in the last 5 years.
- **Churn rate is consistent across departments** — this is a company-wide
  issue, not department-specific.
- **Longer-tenured employees (7+ years) almost never leave**, and their
  salary distribution is not disproportionately high — suggesting loyalty
  is not purely financially driven.

---

## Modeling

Two approaches were implemented and compared.

### Approach A: Logistic Regression
- Outliers in `tenure` removed (IQR method, 824 rows)
- `department` dummy encoded; `salary` ordinally encoded (low=0, medium=1, high=2)
- 75/25 train/test split, stratified

### Approach B: Tree-based Models (Champion)
- Decision Tree and Random Forest with GridSearchCV
- **Round 1:** All features included
- **Round 2 (feature engineering):** `satisfaction_level` dropped (potential
  data leakage — companies may not have this data in real time);
  `average_monthly_hours` replaced with binary `overworked` feature
  (1 if >175 hrs/month, 0 otherwise)

### Model Performance Summary

| Model | Precision | Recall | F1 | Accuracy | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.79 | 0.83 | 0.80 | 0.83 | — |
| Decision Tree Round 1 (CV) | 0.91 | 0.92 | 0.92 | 0.97 | 0.97 |
| Random Forest Round 1 (CV) | 0.95 | 0.92 | 0.93 | 0.98 | 0.98 |
| Decision Tree Round 2 (CV) | 0.86 | 0.90 | 0.88 | 0.96 | 0.96 |
| **Random Forest Round 2 (test)** | **0.87** | **0.90** | **0.89** | **0.96** | **0.94** |

**Champion model:** Random Forest Round 2 — strong performance even after
removing potentially leaky features, confirming the model captures genuine
signal.

### Top Predictive Features
Both the decision tree and random forest agreed on the same top 4 features:
1. `last_evaluation` — performance score
2. `number_project` — workload volume
3. `tenure` — years at company
4. `overworked` — binary flag for >175 hrs/month

---

## Business Recommendations

Based on model results and EDA findings, the following actions are recommended
to Salifort Motors leadership:

1. **Cap project assignments.** No employee should handle more than 5
   projects simultaneously. All 7-project employees churned.
2. **Investigate the 4-year tenure cliff.** Employees at this mark show
   unusually low satisfaction. A policy review or targeted retention
   conversation at the 3.5-year mark could preempt this.
3. **Decouple hours from evaluation scores.** High evaluation scores
   should not require 200+ monthly hours. Recalibrate performance
   metrics to reward efficiency, not just volume.
4. **Create a transparent promotion pipeline.** Very few high-hour
   employees received promotions. Clarity on advancement criteria would
   reduce the sense of inequity.
5. **Conduct workload audits by team.** While churn is consistent across
   departments, average hours are not — targeted interventions by team
   size and project load would be more efficient.

---

## Ethical Considerations

- **Data leakage risk:** `satisfaction_level` may not be available
  in real time for all employees. Round 2 models intentionally exclude
  it to test generalisability.
- **False positive risk:** Employees predicted to churn who are not
  actually at risk may receive unwanted outreach. Interventions should
  be low-stakes (e.g. manager check-ins) rather than consequential
  (e.g. restructuring).
- **Model limitations:** The data shows unusual distributional patterns
  (e.g. sharp clustering in the hours-satisfaction scatterplot) consistent
  with synthetic or manipulated data. Real-world performance may differ.

---

## Tools and Libraries

Python · pandas · numpy · matplotlib · seaborn · scikit-learn
(LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
GridSearchCV) · pickle · PACE framework

---

## About This Project

Completed as the capstone project of the
[Google Advanced Data Analytics Certificate](https://grow.google/certificates/data-analytics/)
(Course 7). All analysis, feature engineering, model interpretation, and
recommendations are original work.

---
