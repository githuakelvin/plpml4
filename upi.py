import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import mean_squared_error, r2_scores
from sklearn.metrics import mean_squared_error, r2_score  #

# Load data
#df = pd.read_csv('breast_cancer.csv')

file_path = r'C:\Users\Administrator\Desktop\global_cancer_patients_2015_2024.csv'
df = pd.read_csv(file_path)

print(df.head())

print(df.columns)

# Check for NaN values
print("NaN values before cleaning:")
print(df.isnull().sum())

# Drop rows with NaN in the target variable
df = df.dropna(subset=['Target_Severity_Score'])

# Clean data
#df.dropna(inplace=True)

# Drop rows with NaN in the target variable
df = df.dropna(subset=['Target_Severity_Score'])
#X = df.drop(['Air_Pollution', 'Target_Severity_Score'], axis=1)  # Features
#y = df['Genetic_Risk']  # Target (M=malignant/B=benign)
# Data Preprocessing
# Drop unnecessary columns (if needed)
X = df.drop(['Patient_ID', 'Target_Severity_Score'], axis=1)  # Features
y = df['Target_Severity_Score']  # Target variable



categorical_cols = ['Gender', 'Country_Region', 'Cancer_Type', 'Cancer_Stage']
label_encoders = {}  # This will store encoders for each column

for col in categorical_cols:
    le = LabelEncoder()  #  Create a new LabelEncoder
    X[col] = le.fit_transform(X[col])  # Fit and transform the column
    label_encoders[col] = le  # Store the encoder for future use
    
    # Final NaN check (in case encoding introduced issues)
print("\nNaN values after encoding:")
print(X.isnull().sum())

# Drop any remaining NaN (if any)
X = X.dropna()
y = y[X.index] 

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2 ,random_state=42, )

# Feature Scaling (optional but can help some models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")

new_data = pd.DataFrame({
    'Age': [50],
    'Gender': ['Female'],
    'Country_Region': ['USA'],
    'Year': [2023],
    'Genetic_Risk': [5],
    'Air_Pollution': [4],
    'Alcohol_Use': [3],
    'Smoking': [2],
    'Obesity_Level': [4],
    'Cancer_Type': ['Lung'],
    'Cancer_Stage': ['Stage II'],
    'Treatment_Cost_USD': [45000],
    'Survival_Years': [3.5]
})

# Preprocess new data (same as training)
for col in categorical_cols:
    le = label_encoders[col]
    new_data[col] = le.transform(new_data[col])

new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"\nPredicted Severity Score: {prediction[0]:.2f}")