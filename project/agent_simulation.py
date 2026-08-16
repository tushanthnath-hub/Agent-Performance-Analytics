import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib


np.random.seed(42)


def generate_agent_data(n=1000):
    agent_id = np.arange(1, n + 1)
    steps_taken = np.random.randint(10, 200, n)
    avg_decision_time = np.random.uniform(0.1, 5.0, n)
    mistakes_made = np.random.randint(0, 20, n)
    environment_type = np.random.choice(['easy', 'medium', 'hard'], n)

    success_prob = (
        0.9
        - (mistakes_made * 0.03)
        - (steps_taken * 0.001)
        - np.where(environment_type == 'hard', 0.2, 0)
        - np.where(environment_type == 'medium', 0.1, 0)
    )

    success = (np.random.rand(n) < success_prob).astype(int)

    df = pd.DataFrame(
        {
            'agent_id': agent_id,
            'steps_taken': steps_taken,
            'avg_decision_time': avg_decision_time,
            'mistakes_made': mistakes_made,
            'environment_type': environment_type,
            'success': success,
        }
    )
    return df


def main():
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / 'data'
    models_dir = project_dir / 'models'

    data_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    df = generate_agent_data(1000)
    df.to_csv(data_dir / 'agent_data.csv', index=False)
    print('Synthetic data saved →', data_dir / 'agent_data.csv')

    X = df.drop(columns=['agent_id', 'success'])
    y = df['success']

    numeric_features = ['steps_taken', 'avg_decision_time', 'mistakes_made']
    categorical_features = ['environment_type']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log_reg_pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression(max_iter=1000)),
        ]
    )
    log_reg_pipeline.fit(X_train, y_train)
    y_pred_lr = log_reg_pipeline.predict(X_test)

    print('\n=== Logistic Regression Results ===')
    print('Accuracy:', accuracy_score(y_test, y_pred_lr))
    print(classification_report(y_test, y_pred_lr))

    rf_pipeline = Pipeline(
        [
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    print('\n=== Random Forest Results ===')
    print('Accuracy:', accuracy_score(y_test, y_pred_rf))
    print(classification_report(y_test, y_pred_rf))

    X_cluster = df[['steps_taken', 'avg_decision_time', 'mistakes_made']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['behavior_cluster'] = kmeans.fit_predict(X_scaled)

    print('\nCluster counts:')
    print(df['behavior_cluster'].value_counts())

    df.to_csv(data_dir / 'agent_data_with_clusters.csv', index=False)
    print('Data with clusters saved →', data_dir / 'agent_data_with_clusters.csv')

    joblib.dump(log_reg_pipeline, models_dir / 'logistic_regression.pkl')
    joblib.dump(rf_pipeline, models_dir / 'random_forest.pkl')
    joblib.dump(kmeans, models_dir / 'kmeans.pkl')
    joblib.dump(scaler, models_dir / 'scaler.pkl')

    print('\nModels saved in', models_dir)


if __name__ == '__main__':
    main()
