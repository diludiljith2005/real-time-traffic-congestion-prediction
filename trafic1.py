import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

st.title("Traffic Volume Prediction Dashboard 🚗🚚🚦")

# 1. Load dataset
df = pd.read_csv('TrafficTwoMonth.csv')
st.subheader("Dataset Preview")
st.dataframe(df.head())

# 2. Basic info
with st.expander("Basic Info"):
    st.text(df.info())
    st.write(df.describe())

# 3. Missing values
st.write("### Missing Values")
st.write(df.isnull().sum())

# 4. Create datetime column
try:
    df['DateTime'] = pd.to_datetime('2025-05-' + df['Date'].astype(str).str.zfill(2) + ' ' + df['Time'])
except Exception as e:
    st.error(f"Error parsing DateTime: {e}")
    st.stop()

df.set_index('DateTime', inplace=True)

# 5. Visualize total traffic over time
st.write("### Total Traffic Over Time")
fig, ax = plt.subplots(figsize=(15,5))
df['Total'].plot(ax=ax)
ax.set_xlabel('DateTime')
ax.set_ylabel('Total Traffic')
st.pyplot(fig)

# 6. Encode categorical variables
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# 7. Fill missing values
df.fillna(df.mean(), inplace=True)

# 8. Define X and y
X = df.drop('Total', axis=1)
y = df['Total']

# 9. Train/Test Split
test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=20, step=5)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42)

st.write(f"Train size: {X_train.shape[0]} samples")
st.write(f"Test size: {X_test.shape[0]} samples")

# 10. Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 11. Predict
y_pred = model.predict(X_test)

# 12. Evaluate
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
accuracy = r2_score(y_test, y_pred) * 100  # Convert to percentage

st.write("### Model Evaluation")
st.write(f"MSE: {mse:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"MAE: {mae:.2f}")
st.write(f"Accuracy (R² Score): {accuracy:.2f}%")

# 13. Plot original & smoothed traffic
df['Total_smooth'] = df['Total'].rolling(window=10).mean()
st.write("### Smoothed Traffic Volume")
fig2, ax2 = plt.subplots(figsize=(15,5))
ax2.plot(df.index, df['Total'], label='Original')
ax2.plot(df.index, df['Total_smooth'], color='red', label='Smoothed (window=10)')
ax2.set_xlabel('DateTime')
ax2.set_ylabel('Total Traffic')
ax2.legend()
st.pyplot(fig2)
