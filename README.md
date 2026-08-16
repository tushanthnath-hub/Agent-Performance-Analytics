# Agent_Performance_Analytics


# 🤖 AI Agent Performance Analytics & Behavior Prediction

An end-to-end machine learning project for analyzing AI agent behavior, predicting task success, and discovering behavioral patterns using classification and clustering.

The project generates synthetic agent interaction data, preprocesses the data, trains multiple machine learning models, performs behavior clustering, and provides an interactive Streamlit dashboard for visualization and prediction.

---

## 🚀 Project Overview

AI agents can behave differently depending on factors such as:

* Number of steps taken
* Decision-making time
* Number of mistakes
* Environment difficulty

This project uses these behavioral signals to:

1. Generate synthetic agent performance data
2. Clean and preprocess the dataset
3. Predict whether an agent will successfully complete a task
4. Compare Logistic Regression and Random Forest models
5. Cluster agents based on behavioral patterns
6. Visualize results through an interactive dashboard

---

## 🧠 Machine Learning Techniques

### Classification

Two models are trained and compared:

* Logistic Regression
* Random Forest Classifier

### Clustering

K-Means clustering is used to identify different agent behavior patterns.

### Preprocessing

The ML pipeline includes:

* Feature scaling using `StandardScaler`
* Categorical encoding using `OneHotEncoder`
* Train/test split
* Reusable Scikit-learn pipelines

---

## 📊 Features

The dataset contains the following features:

| Feature             | Description                                       |
| ------------------- | ------------------------------------------------- |
| `agent_id`          | Unique agent identifier                           |
| `steps_taken`       | Number of steps taken to complete a task          |
| `avg_decision_time` | Average time taken for decisions                  |
| `mistakes_made`     | Number of mistakes made                           |
| `environment_type`  | Easy, medium, or hard environment                 |
| `success`           | Whether the agent successfully completed the task |
| `behavior_cluster`  | Cluster assigned by K-Means                       |

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

---

## 📈 Machine Learning Workflow

```text
Synthetic Data
      ↓
Data Preprocessing
      ↓
Train/Test Split
      ↓
 ┌───────────────┐
 │ Classification│
 └───────────────┘
      ↓
Logistic Regression
      +
Random Forest
      ↓
Model Evaluation
      ↓
Best Model
      ↓
Prediction
```

Behavior analysis follows a separate pipeline:

```text
Agent Behavioral Data
        ↓
Feature Scaling
        ↓
K-Means Clustering
        ↓
Behavior Groups
        ↓
Dashboard Visualization
```

---

## 🔍 Example Questions the Project Can Answer

* Which factors are associated with successful agents?
* Does environment difficulty affect success rate?
* Do agents that take more steps make more mistakes?
* Which behavioral cluster contains the most successful agents?
* Which ML model performs better?
* Which features are most important for predicting success?

---

## 📊 Dashboard

The Streamlit dashboard provides interactive visualizations for:

### Agent Behavior Clusters

Visualizes agents based on steps taken and decision-making time.

### Environment Performance

Compares success rates across:

* Easy
* Medium
* Hard

### Agent Success Prediction

Users can enter:

* Steps taken
* Average decision time
* Number of mistakes
* Environment type

and receive a predicted outcome.

---

## 🔮 Future Improvements

Potential improvements include:

* [ ] Add XGBoost/Gradient Boosting
* [ ] Add ROC-AUC visualization
* [ ] Add confusion matrix visualization
* [ ] Add model feature importance
* [ ] Add cross-validation
* [ ] Improve K-Means cluster interpretation
* [ ] Add automated model comparison
* [ ] Add experiment tracking
* [ ] Add unit tests
* [ ] Add Docker support
* [ ] Deploy the Streamlit dashboard
* [ ] Replace synthetic data with real agent evaluation data

---


## 🎯 Learning Objectives

This project demonstrates practical experience with:

* Machine learning pipelines
* Classification
* Unsupervised learning
* Feature preprocessing
* Model evaluation
* Data visualization
* Model persistence
* Streamlit application development
* Basic ML project organization

