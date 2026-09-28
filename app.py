import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Regression model values
SLOPE = 0.47987
INTERCEPT = 73.82903

st.set_page_config(
    page_title="Exercise & Resting Heart Rate",
    page_icon="🏃",
    layout="wide"
)

st.title("🏃 Weekly Exercise & Resting Heart Rate")
st.write(
    "Statistical correlation and predictive regression analysis "
    "based on survey data."
)

st.divider()

# Prediction section
st.subheader("🔮 Resting Heart Rate Prediction")

exercise_hours = st.number_input(
    "Weekly Exercise / Physical Activity Hours",
    min_value=0.0,
    max_value=30.0,
    value=5.0,
    step=0.5
)

if st.button("Predict Resting Heart Rate"):
    predicted_hr = INTERCEPT + SLOPE * exercise_hours
    st.success(
        f"Predicted Resting Heart Rate: {predicted_hr:.2f} BPM"
    )

st.divider()

# Regression model
st.subheader("📈 Regression Model")

st.write("Ŷ = 73.82903 + 0.47987 × X")
st.write("X = Weekly Exercise Hours")
st.write("Ŷ = Predicted Resting Heart Rate")

st.divider()

# Statistical results
st.subheader("📊 Statistical Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Pearson Correlation (r)", "0.13254")

with col2:
    st.metric("Spearman Correlation (ρ)", "0.17284")

with col3:
    st.metric("R²", "0.01757")

col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Slope (b₁)", "0.47987")

with col5:
    st.metric("Intercept (b₀)", "73.82903")

with col6:
    st.metric("RMSE", "13.91766")

# Final dataset
data = {
    "exercise_hours": [
        1, 3, 7, 3, 10, 7, 3, 5, 2, 3,
        5, 5, 2, 15, 12, 1, 1, 1, 3, 2,
        7, 5, 4, 6, 4, 1, 3, 3, 5, 2,
        10, 10, 6, 5, 5, 6, 6, 6, 8, 14,
        6, 17, 5, 7, 14, 2, 1, 4
    ],
    "resting_hr": [
        65, 70, 78, 80, 82, 92, 77, 80, 115, 70,
        85, 80, 65, 65, 129, 76, 69, 70, 80, 69,
        105, 80, 70, 80, 80, 86, 70, 65, 60, 70,
        75, 80, 74, 65, 70, 98, 100, 70, 60, 60,
        70, 75, 60, 70, 80, 60, 80, 60
    ]
}

df = pd.DataFrame(data)

# Predictions and residuals
df["predicted_hr"] = (
    INTERCEPT + SLOPE * df["exercise_hours"]
)

df["residual"] = (
    df["resting_hr"] - df["predicted_hr"]
)

# Scatter plot with regression line
st.subheader("📉 Exercise Hours vs Resting Heart Rate")

fig, ax = plt.subplots()

ax.scatter(
    df["exercise_hours"],
    df["resting_hr"],
    label="Survey Data"
)

x_line = df["exercise_hours"]
y_line = INTERCEPT + SLOPE * x_line

ax.plot(
    x_line,
    y_line,
    label="Regression Line"
)

ax.set_xlabel("Weekly Exercise Hours")
ax.set_ylabel("Resting Heart Rate (BPM)")
ax.set_title("Exercise Hours vs Resting Heart Rate")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Residual plot
st.subheader("📐 Residual Plot")

fig2, ax2 = plt.subplots()

ax2.scatter(
    df["predicted_hr"],
    df["residual"]
)

ax2.axhline(
    0,
    linestyle="--"
)

ax2.set_xlabel("Predicted Resting Heart Rate (BPM)")
ax2.set_ylabel("Residual")
ax2.set_title("Residual Plot")
ax2.grid(True)

st.pyplot(fig2)

# Accuracy metrics
st.subheader("🎯 Model Accuracy")

actual = df["resting_hr"]
predicted = df["predicted_hr"]

errors = actual - predicted

mae = errors.abs().mean()

rmse = (
    errors.pow(2).mean()
) ** 0.5

sst = (
    (actual - actual.mean()) ** 2
).sum()

sse = (
    errors ** 2
).sum()

r2 = 1 - (sse / sst)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.5f}")

with col2:
    st.metric("RMSE", f"{rmse:.5f}")

with col3:
    st.metric("R²", f"{r2:.5f}")

# Dataset table
st.subheader("📋 Final Dataset")

st.dataframe(
    df[
        [
            "exercise_hours",
            "resting_hr",
            "predicted_hr",
            "residual"
        ]
    ],
    use_container_width=True
)

# About model
st.subheader("ℹ️ About the Model")

st.write(
    "The model uses simple linear regression to estimate "
    "resting heart rate from weekly exercise hours."
)

st.write(
    "The regression coefficients were calculated from "
    "the collected survey dataset."
)

st.info(
    "This prediction is for academic and statistical analysis "
    "purposes only. It should not be used as medical advice "
    "or for diagnosis."
)
