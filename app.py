"""Customer churn risk explorer.

Run locally:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
            "master/data/Telco-Customer-Churn.csv")
NUM = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

st.set_page_config(page_title="Churn Risk Explorer", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_resource
def train(df):
    X = df.drop(columns=["customerID", "Churn"])
    cat = [c for c in X.columns if c not in NUM]
    pre = ColumnTransformer([("num", StandardScaler(), NUM),
                             ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat)])
    model = Pipeline([("pre", pre), ("clf", XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42))])
    model.fit(X, df["Churn"])
    explainer = shap.TreeExplainer(model.named_steps["clf"])
    names = [n.split("__", 1)[1] for n in model.named_steps["pre"].get_feature_names_out()]
    return model, explainer, names, X.columns.tolist()


df = load_data()
model, explainer, feat_names, input_cols = train(df)
options = {c: sorted(df[c].unique().tolist()) for c in input_cols if c not in NUM}

st.title("Customer Churn Risk Explorer")
st.caption("XGBoost model trained on 7,043 telecom customers (IBM Telco dataset). "
           "Change a customer's details in the sidebar to see how their churn risk moves, and why.")

# ---- inputs -------------------------------------------------------------
sb = st.sidebar
sb.header("Customer profile")
tenure = sb.slider("Tenure (months)", 0, 72, 5)
monthly = sb.slider("Monthly charges ($)", 18.0, 120.0, 85.0, step=0.5)
contract = sb.selectbox("Contract", options["Contract"], index=options["Contract"].index("Month-to-month"))
internet = sb.selectbox("Internet service", options["InternetService"],
                        index=options["InternetService"].index("Fiber optic"))
payment = sb.selectbox("Payment method", options["PaymentMethod"],
                       index=options["PaymentMethod"].index("Electronic check"))
security = sb.selectbox("Online security", options["OnlineSecurity"], index=0)
support = sb.selectbox("Tech support", options["TechSupport"], index=0)
paperless = sb.selectbox("Paperless billing", options["PaperlessBilling"], index=1)
senior = sb.checkbox("Senior citizen", value=False)

# Fill the remaining fields with the most common value in the data
row = {c: df[c].mode()[0] for c in input_cols}
row.update({"tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
            "Contract": contract, "InternetService": internet, "PaymentMethod": payment,
            "OnlineSecurity": security, "TechSupport": support, "PaperlessBilling": paperless,
            "SeniorCitizen": int(senior)})
if internet == "No":
    for c in ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
        row[c] = "No internet service"
x = pd.DataFrame([row])[input_cols]

# ---- prediction ---------------------------------------------------------
p = float(model.predict_proba(x)[0, 1])
base_rate = df["Churn"].mean()
band = "High" if p >= 0.5 else "Medium" if p >= 0.25 else "Low"

c1, c2, c3 = st.columns(3)
c1.metric("Churn probability", f"{p:.0%}", f"{(p - base_rate) * 100:+.0f} pts vs average customer",
          delta_color="inverse")
c2.metric("Risk band", band)
c3.metric("Annual revenue at risk", f"${12 * monthly * p:,.0f}")

# ---- explanation --------------------------------------------------------
xt = model.named_steps["pre"].transform(x)
xt = xt.toarray() if hasattr(xt, "toarray") else xt
sv = explainer.shap_values(xt)[0]
contrib = pd.Series(sv, index=feat_names)
contrib = contrib[contrib.abs() > 0.01].sort_values(key=np.abs, ascending=False).head(10)

left, right = st.columns([3, 2])
with left:
    st.subheader("What drives this prediction")
    fig, ax = plt.subplots(figsize=(6, 4))
    contrib.sort_values().plot.barh(ax=ax, color=["#2e7d32" if v < 0 else "#c62828" for v in contrib.sort_values()])
    ax.set_xlabel("effect on churn score (SHAP, log-odds)")
    ax.axvline(0, color="grey", lw=0.8)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Red bars push risk up, green bars push it down.")

with right:
    st.subheader("Suggested retention actions")
    actions = []
    if contract == "Month-to-month":
        actions.append("Offer a discount for switching to a one- or two-year contract.")
    if payment == "Electronic check":
        actions.append("Offer a small credit for moving to automatic payment.")
    if security == "No" or support == "No":
        actions.append("Bundle online security or tech support free for 3 months.")
    if tenure < 12:
        actions.append("Enroll in a first-year onboarding check-in.")
    if monthly > 80:
        actions.append("Review the plan for a lower-cost option with the same services.")
    if not actions:
        actions.append("No specific action needed; standard engagement.")
    for a in actions:
        st.markdown(f"- {a}")

st.divider()
st.caption("Built by Muhammad Ehtesham · Model and analysis: "
           "[github.com/mehtesham13/customer-churn-prediction](https://github.com/mehtesham13/customer-churn-prediction)")
