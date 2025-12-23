import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Page config
st.set_page_config("Regression Models", layout="centered")

# Load CSS
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# Sidebar
st.sidebar.title("Model Settings")

model_type = st.sidebar.selectbox(
    "Select Regression Model",
    ["Simple Linear Regression", "Multiple Linear Regression", "Polynomial Regression"]
)

poly_degree = 2
if model_type == "Polynomial Regression":
    poly_degree = st.sidebar.slider("Polynomial Degree", 2, 5, 2)

# Title
st.markdown(f"""
<div class="card">
<h1>{model_type}</h1>
<p>Predict <b>Tip Amount</b> from restaurant data</p>
</div>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# Dataset Preview
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# Feature Selection
if model_type == "Simple Linear Regression":
    X = df[["total_bill"]]
    y = df["tip"]

elif model_type == "Multiple Linear Regression":
    X = df[["total_bill", "size"]]
    y = df["tip"]

else:
    X = df[["total_bill"]]
    y = df["tip"]

# Train-Test Split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

# Model Training
if model_type == "Polynomial Regression":
    poly = PolynomialFeatures(degree=poly_degree)
    X_tr_final = poly.fit_transform(X_tr_scaled)
    X_te_final = poly.transform(X_te_scaled)
else:
    X_tr_final = X_tr_scaled
    X_te_final = X_te_scaled

model = LinearRegression()
model.fit(X_tr_final, y_tr)
y_pred = model.predict(X_te_final)

# Metrics
mae = mean_absolute_error(y_te, y_pred)
mse = mean_squared_error(y_te, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_te, y_pred)
adj_r2 = 1 - (1 - r2) * (len(y_te) - 1) / (len(y_te) - X.shape[1] - 1)


# Visualization
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Visualization")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

x_range = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100)

if model_type == "Multiple Linear Regression":
    mean_size = df["size"].mean()
    X_plot = np.column_stack((x_range, np.full_like(x_range, mean_size)))
    X_plot_scaled = scaler.transform(X_plot)
    y_plot = model.predict(X_plot_scaled)

else:
    x_range = x_range.reshape(-1, 1)
    x_scaled = scaler.transform(x_range)

    if model_type == "Polynomial Regression":
        x_scaled = poly.transform(x_scaled)

    y_plot = model.predict(x_scaled)

ax.plot(x_range, y_plot, color="red")
ax.set_xlabel("Total Bill")
ax.set_ylabel("Tip")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)


# Performance
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# Coefficients
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Intercept & Co-efficient")

if model_type == "Polynomial Regression":
    st.write(f"Polynomial Degree: {poly_degree}")
    st.write("Coefficients:")
    st.write(model.coef_)
else:
    for col, coef in zip(X.columns, model.coef_):
        st.write(f"{col}: {coef:.3f}")

st.write(f"Intercept: {model.intercept_:.3f}")
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

if model_type == "Multiple Linear Regression":
    bill = st.slider("Total Bill ($)", float(df.total_bill.min()), float(df.total_bill.max()), 30.0)
    
    size = st.slider(
    "Group Size",
    int(df["size"].min()),
    int(df["size"].max()),
    2
)

    input_data = scaler.transform([[bill, size]])
else:
    bill = st.slider("Total Bill ($)", float(df.total_bill.min()), float(df.total_bill.max()), 30.0)
    input_data = scaler.transform([[bill]])

if model_type == "Polynomial Regression":
    input_data = poly.transform(input_data)

prediction = model.predict(input_data)[0]

st.markdown(
    f'<div class="prediction-box">Predict Tip: ${prediction:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
