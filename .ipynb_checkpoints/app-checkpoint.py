import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Data/Crop_recommendation.csv")

# Remove the label column for clustering
X_cluster = df.drop(['label'], axis=1).values

# Perform KMeans clustering
km = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_means = km.fit_predict(X_cluster)

# Add clustering results to the dataframe
df['Cluster'] = y_means

# Split the dataset for training and validation
X = df.drop(['label', 'Cluster'], axis=1)
y = df['label']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a Logistic Regression model
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)

# Set page style
st.set_page_config(page_title='🌾 Crop Recommendation App', page_icon='🌱', layout='centered')

# Streamlit app
st.title('🌾 Crop Recommendation App 🌱')

# Two columns for sliders
col1, col2 = st.columns(2)

# User input for N, P, K
with col1:
    N = st.slider('🌾 Nitrogen (N)', min_value=0, max_value=140, value=50)
    P = st.slider('🌾 Phosphorus (P)', min_value=5, max_value=150, value=50)
    K = st.slider('🌾 Potassium (K)', min_value=5, max_value=210, value=50)

# User input for Temp, Humidity, pH, Rainfall
with col2:
    Temp = st.slider('🌡️ Temperature', min_value=8, max_value=45, value=50)
    Humidity = st.slider('💧 Humidity', min_value=14, max_value=100, value=50)
    pH = st.slider('📈 pH Level', min_value=3, max_value=10, value=50)
    Rainfall = st.slider('🌧️ Rainfall (mm)', min_value=20, max_value=300, value=50)

# Custom styled button with CSS
button_style = """
    <style>
        div[data-testid="stButton"] button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
            border-radius: 8px;
        }
    </style>
"""

st.markdown(button_style, unsafe_allow_html=True)

# Prediction button
if st.button('🚀 Predict Crop'):
    # Make prediction
    pred = model.predict([[N, P, K, Temp, Humidity, pH, Rainfall]])

    # Display predicted crop
    st.subheader('🌱 Predicted Crop:')
    st.write(f"The suggested crop for given climatic conditions is: **{pred[0]}**")

    # Display a single frequency distribution line graph for all conditions
    st.subheader('📈 Distribution of Conditions for the Predicted Crop:')

    # Filter dataframe based on predicted crop
    crop_df = df[df['label'] == pred[0]]

    # Plot histograms for each condition
    conditions = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    plt.figure(figsize=(12, 8))

    for condition in conditions:
        sns.histplot(crop_df[condition], label=condition, kde=True, element="step", stat="density", common_norm=False)

    plt.title(f'Distribution of Conditions for {pred[0]}')
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.legend()
    st.pyplot(plt)
