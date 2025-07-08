import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Cancer Severity Predictor", layout="wide")

# Title and description
st.title("Cancer Patient Severity Score Predictor")
st.write("""
This app predicts the severity score of cancer patients based on their clinical and demographic characteristics.
Upload your dataset or use the sample data, then make predictions using the trained model.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Features")

# Function to load sample data
@st.cache_data
def load_sample_data():
    sample_data = {
        'Patient_ID': ['PT0000000', 'PT0000001', 'PT0000002'],
        'Age': [71, 34, 80],
        'Gender': ['Male', 'Male', 'Male'],
        'Country_Region': ['UK', 'China', 'Pakistan'],
        'Year': [2020, 2021, 2022],
        'Genetic_Risk': [5, 3, 4],
        'Air_Pollution': [4, 2, 5],
        'Alcohol_Use': [3, 1, 4],
        'Smoking': [4, 1, 5],
        'Obesity_Level': [3, 2, 4],
        'Cancer_Type': ['Lung', 'Breast', 'Prostate'],
        'Cancer_Stage': ['Stage III', 'Stage 0', 'Stage II'],
        'Treatment_Cost_USD': [62913.44, 12573.41, 6984.33],
        'Survival_Years': [5.9, 4.7, 7.1],
        'Target_Severity_Score': [4.92, 4.65, 5.84]
    }
    return pd.DataFrame(sample_data)

# File upload or use sample data
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_sample_data()
    st.sidebar.info("Using sample data. Upload a CSV file to use your own data.")

# Display raw data
st.subheader("Raw Data")
st.write(df)

# Data preprocessing
st.subheader("Data Preprocessing")

# Show missing values
st.write("Missing values in dataset:")
st.write(df.isnull().sum())

# Drop rows with missing target values
df = df.dropna(subset=['Target_Severity_Score'])

# Features and target
X = df.drop(['Patient_ID', 'Target_Severity_Score'], axis=1)
y = df['Target_Severity_Score']

# Encode categorical variables
categorical_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Drop any remaining NaN
X = X.dropna()
y = y[X.index]

# Model training section
st.subheader("Model Training")

# Split data
test_size = st.slider("Select test set size (%)", 10, 40, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
n_estimators = st.slider("Number of trees in the forest", 50, 500, 100)
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"Model Performance:")
st.write(f"- Mean Squared Error: {mse:.2f}")
st.write(f"- R² Score: {r2:.2f}")

# Feature importance
st.subheader("Feature Importance")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
ax.set_title('Feature Importance')
st.pyplot(fig)

# Prediction section
st.subheader("Make New Predictions")

# Create input form
with st.form("prediction_form"):
    st.write("Enter patient details:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
        country = st.selectbox("Country", ['USA', 'UK', 'China', 'Pakistan', 'Brazil'])
        year = st.number_input("Year", min_value=2015, max_value=2024, value=2023)
        genetic_risk = st.slider("Genetic Risk (1-10)", 1, 10, 5)
        air_pollution = st.slider("Air Pollution Exposure (1-10)", 1, 10, 4)
        
    with col2:
        alcohol_use = st.slider("Alcohol Use (1-10)", 1, 10, 3)
        smoking = st.slider("Smoking (1-10)", 1, 10, 2)
        obesity = st.slider("Obesity Level (1-10)", 1, 10, 4)
        cancer_type = st.selectbox("Cancer Type", ['Lung', 'Breast', 'Prostate', 'Colon', 'Liver'])
        cancer_stage = st.selectbox("Cancer Stage", ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'])
        treatment_cost = st.number_input("Treatment Cost (USD)", min_value=0, value=45000)
        survival_years = st.number_input("Expected Survival Years", min_value=0.0, value=3.5, step=0.1)
    
    submitted = st.form_submit_button("Predict Severity Score")

if submitted:
    # Create input dataframe
    new_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Country_Region': [country],
        'Year': [year],
        'Genetic_Risk': [genetic_risk],
        'Air_Pollution': [air_pollution],
        'Alcohol_Use': [alcohol_use],
        'Smoking': [smoking],
        'Obesity_Level': [obesity],
        'Cancer_Type': [cancer_type],
        'Cancer_Stage': [cancer_stage],
        'Treatment_Cost_USD': [treatment_cost],
        'Survival_Years': [survival_years]
    })
    
    # Preprocess input data
    for col in categorical_cols:
        le = label_encoders[col]
        # Handle unseen labels by assigning a default value
        try:
            new_data[col] = le.transform(new_data[col])
        except ValueError:
            new_data[col] = 0  # Assign to first category if label not seen before
    
    # Scale features
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)
    
    # Display result
    st.success(f"Predicted Severity Score: {prediction[0]:.2f}")
    
    # Interpretation
    st.write("""
    **Severity Score Interpretation:**
    - 1-2: Very Low Severity
    - 2-3: Low Severity
    - 3-4: Moderate Severity
    - 4-5: High Severity
    - 5-6: Very High Severity
    """)

# Data visualization
st.subheader("Data Visualization")

# Select variable to plot against target
plot_var = st.selectbox("Select variable to plot against Severity Score", 
                       ['Age', 'Genetic_Risk', 'Air_Pollution', 'Alcohol_Use', 
                        'Smoking', 'Obesity_Level', 'Treatment_Cost_USD', 'Survival_Years'])

fig2, ax2 = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=plot_var, y='Target_Severity_Score', data=df, ax=ax2)
ax2.set_title(f'{plot_var} vs. Severity Score')
st.pyplot(fig2)