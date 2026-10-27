import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.title("CSV Explorer & Linear Regression")

# Step 3 of the assignment: one sentence answering the question.
# TODO: rewrite this for YOUR dataset.
st.markdown(
    "**About this model:** It gives a rough estimate of a movie's box-office "
    "revenue from its budget and number of audience votes, which could help "
    "compare films at a glance; when it is wrong, a studio or investor could "
    "badly over- or underestimate a film's earnings, so treat it as a ballpark, "
    "not a forecast."
)

# 1. Upload + preview
uploaded = st.file_uploader("Upload a CSV file", type="csv")
if uploaded is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Preview")
st.dataframe(df.head())

# ID columns are numbers but not meaningful measurements, so leave them out
numeric_cols = [
    c for c in df.select_dtypes("number").columns
    if c.lower() not in ("id", "movie_id")
]

# In the TMDB data, a budget or revenue of 0 means "unknown", not zero dollars
if st.checkbox("Treat 0 as missing (recommended for budget/revenue)", value=True):
    for c in ("budget", "revenue"):
        if c in df.columns:
            df[c] = df[c].astype("float").replace(0, float("nan"))
    st.caption("Rows with 0 budget or revenue are ignored in charts and the model.")

if len(numeric_cols) < 3:
    st.error("This app needs a CSV with at least three numeric columns.")
    st.stop()

# 2. Column + chart type
st.subheader("Chart")
chart_type = st.selectbox("Chart type", ["Line", "Scatter", "Histogram"])
col = st.selectbox("Column", numeric_cols)

fig, ax = plt.subplots()
if chart_type == "Line":
    ax.plot(df[col].values)
    ax.set_xlabel("Row")
    ax.set_ylabel(col)
elif chart_type == "Scatter":
    x_col = st.selectbox("X-axis column", [c for c in numeric_cols if c != col])
    ax.scatter(df[x_col], df[col], alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(col)
else:
    ax.hist(df[col].dropna(), bins=30)
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
st.pyplot(fig)

# 3. Linear regression with two inputs
st.subheader("Linear regression")
# Default to the model described in the sentence at the top (if those columns exist)
default_target = "revenue" if "revenue" in numeric_cols else numeric_cols[-1]
target = st.selectbox(
    "Target (what to predict)", numeric_cols,
    index=numeric_cols.index(default_target),
)
feature_options = [c for c in numeric_cols if c != target]
preferred = [c for c in ("budget", "vote_count") if c in feature_options]
default_features = preferred if len(preferred) == 2 else feature_options[:2]
features = st.multiselect(
    "Choose exactly two inputs", feature_options,
    default=default_features, max_selections=2,
)
if len(features) != 2:
    st.warning("Please choose exactly two inputs.")
    st.stop()

data = df[features + [target]].dropna()
X = data[features]
y = data[target]
model = LinearRegression().fit(X, y)
st.write(f"R² on the training data: {model.score(X, y):.3f}")

st.markdown("**Enter values to get a prediction:**")
inputs = {f: st.number_input(f, value=float(X[f].mean())) for f in features}

# Hint from the assignment: same column names and order as in training
new_X = pd.DataFrame([inputs], columns=features)
prediction = model.predict(new_X)[0]
st.success(f"Predicted {target}: {prediction:,.2f}")
