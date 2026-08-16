from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st


project_dir = Path(__file__).resolve().parent
data_path = project_dir / 'data' / 'agent_data_with_clusters.csv'
model_path = project_dir / 'models' / 'random_forest.pkl'

st.title('🤖 Agent Performance Dashboard')

if not data_path.exists():
    st.error('Run agent_simulation.py first to generate the dataset and model files.', icon='⚠️')
    st.stop()

if not model_path.exists():
    st.error('Model file not found. Run agent_simulation.py to train and save the model.', icon='⚠️')
    st.stop()


df = pd.read_csv(data_path)

st.subheader('📌 Dataset Preview')
st.dataframe(df.head())

st.subheader('🔍 Agent Behavior Clusters')
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x=df['steps_taken'],
    y=df['avg_decision_time'],
    hue=df['behavior_cluster'],
    palette='deep',
    ax=ax,
)
ax.set_title('Agent Behavior Clusters')
ax.set_xlabel('Steps Taken')
ax.set_ylabel('Average Decision Time')
st.pyplot(fig)

st.subheader('📊 Success Rate by Environment Type')
success_env = df.groupby('environment_type')['success'].mean().reset_index()
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.barplot(data=success_env, x='environment_type', y='success', palette='viridis', ax=ax2)
ax2.set_title('Success Rate by Environment')
ax2.set_xlabel('Environment Type')
ax2.set_ylabel('Success Rate')
st.pyplot(fig2)

model = joblib.load(model_path)

st.subheader('🧪 Predict Agent Success')
steps = st.slider('Steps Taken', 10, 200, 50)
decision_time = st.slider('Average Decision Time', 0.1, 5.0, 1.5)
mistakes = st.slider('Mistakes Made', 0, 20, 2)
env = st.selectbox('Environment Type', ['easy', 'medium', 'hard'])

if st.button('Predict'):
    sample = pd.DataFrame(
        [{
            'steps_taken': steps,
            'avg_decision_time': decision_time,
            'mistakes_made': mistakes,
            'environment_type': env,
        }]
    )

    pred = model.predict(sample)[0]
    if pred == 1:
        st.success('Agent will SUCCEED')
    else:
        st.error(' Agent will FAIL')
