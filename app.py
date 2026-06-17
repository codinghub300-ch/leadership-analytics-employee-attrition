import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Leadership Analytics Dashboard",
    layout="wide"
)

st.title("🧠 Leadership Analytics & Employee Attrition Dashboard")

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv("data.csv")

# ==========================
# ENCODING
# ==========================
label_cols = [
    'Attrition',
    'BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
    'Over18',
    'OverTime'
]

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ==========================
# FEATURE ENGINEERING
# ==========================
df['EmployeeEngagement'] = (
    df['JobInvolvement'] +
    df['WorkLifeBalance']
) / 2

df['EmployeeSatisfaction'] = (
    df['JobSatisfaction'] +
    df['EnvironmentSatisfaction'] +
    df['RelationshipSatisfaction']
) / 3

df['GrowthRisk'] = (
    df['YearsSinceLastPromotion'] +
    df['YearsInCurrentRole']
)

df['StressScore'] = (
    df['OverTime'] +
    (5 - df['WorkLifeBalance'])
)

# ==========================
# MANAGER ID
# ==========================
df['ManagerID'] = (
    df['Department'].astype(str)
    + "_"
    + df['JobRole'].astype(str)
    + "_"
    + (df['JobLevel']//2).astype(str)
)

# ==========================
# FEATURES
# ==========================
features = [
    'EmployeeEngagement',
    'EmployeeSatisfaction',
    'GrowthRisk',
    'StressScore',
    'MonthlyIncome',
    'DistanceFromHome',
    'TotalWorkingYears'
]

X = df[features]
y = df['Attrition']

# ==========================
# SCALING
# ==========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==========================
# TRAIN TEST SPLIT
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# SIDEBAR
# ==========================
st.sidebar.title("⚙ Model Selection")

model_name = st.sidebar.selectbox(
    "Choose Model",
    [
        "Random Forest",
        "SVM",
        "Logistic + Text Analytics"
    ]
)

# ==========================
# MODEL
# ==========================
if model_name == "Random Forest":

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        random_state=42
    )

elif model_name == "SVM":

    model = SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )

else:

    model = LogisticRegression(
        max_iter=500,
        solver='liblinear',
        random_state=42
    )

# ==========================
# TRAIN MODEL
# ==========================
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

accuracy = accuracy_score(y_test, y_pred)

roc = roc_auc_score(y_test, y_prob)

# ==========================
# PERFORMANCE
# ==========================
st.header("📈 Model Performance")

c1, c2 = st.columns(2)

with c1:
    st.metric(
        "Accuracy",
        f"{accuracy:.3f}"
    )

with c2:
    st.metric(
        "ROC-AUC",
        f"{roc:.3f}"
    )

# ==========================
# ATTRITION RISK
# ==========================
df["AttritionRisk"] = model.predict_proba(X_scaled)[:,1]
# ===========================
# Train Models
# ===========================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    random_state=42
)

svm_model = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)

log_model = LogisticRegression(
    max_iter=500,
    solver='liblinear',
    random_state=42
)

rf_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)
log_model.fit(X_train, y_train)

# ===========================
# Model Selection
# ===========================
# st.sidebar.title("⚙️ Model")

# selected_model = st.sidebar.selectbox(
#     "Choose Model",
#     [
#         "Random Forest",
#         "SVM",
#         "Logistic + Text"
#     ]
# )

# if selected_model == "Random Forest":
#     model = rf_model

# elif selected_model == "SVM":
#     model = svm_model

# else:
#     model = log_model

# ===========================
# Evaluation
# ===========================
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, model.predict_proba(X_test)[:,1])

col1,col2 = st.columns(2)

with col1:
    st.metric("Accuracy",f"{acc:.2%}")

with col2:
    st.metric("ROC-AUC",f"{auc:.2%}")

# ===========================
# Attrition Risk
# ===========================
df["AttritionRisk"] = model.predict_proba(X_scaled)[:,1]

leader_df = df.groupby("ManagerID").agg({
    "AttritionRisk":"mean",
    "EmployeeEngagement":"mean",
    "EmployeeSatisfaction":"mean",
    "StressScore":"mean"
}).reset_index()

leader_df["LeaderScore"] = (
    (1-leader_df["AttritionRisk"])*0.4
    + leader_df["EmployeeEngagement"]*0.2
    + leader_df["EmployeeSatisfaction"]*0.2
    + (1-leader_df["StressScore"]/5)*0.2
)

threshold = leader_df["LeaderScore"].quantile(0.6)

leader_df["LeaderQuality"] = np.where(
    leader_df["LeaderScore"]>=threshold,
    "Good",
    "Not Good"
)

leader_df = leader_df.sort_values(
    by="LeaderScore",
    ascending=False
)

# ===========================
# Top Leaders
# ===========================
st.subheader("🏆 Top 10 Leaders")

st.dataframe(
    leader_df.head(10),
    use_container_width=True
)

# ===========================
# Chart
# ===========================
st.subheader("📊 Leader Scores")

fig, ax = plt.subplots(figsize=(12,5))

colors = [
    "green" if x=="Good" else "red"
    for x in leader_df.head(15)["LeaderQuality"]
]

ax.bar(
    leader_df.head(15)["ManagerID"],
    leader_df.head(15)["LeaderScore"],
    color=colors
)

plt.xticks(rotation=45)

st.pyplot(fig)

# ===========================
# Prediction Section
# ===========================
st.subheader("🔮 Predict Employee Attrition")

engagement = st.slider(
    "Employee Engagement",
    1.0,
    5.0,
    3.0
)

satisfaction = st.slider(
    "Employee Satisfaction",
    1.0,
    5.0,
    3.0
)

growth = st.slider(
    "Growth Risk",
    0,
    30,
    5
)

stress = st.slider(
    "Stress Score",
    0,
    5,
    2
)

income = st.number_input(
    "Monthly Income",
    value=5000
)

distance = st.number_input(
    "Distance From Home",
    value=5
)

years = st.number_input(
    "Total Working Years",
    value=10
)

if st.button("Predict"):

    new_data = pd.DataFrame([[
        engagement,
        satisfaction,
        growth,
        stress,
        income,
        distance,
        years
    ]], columns=features)

    new_scaled = scaler.transform(new_data)

    pred = model.predict(new_scaled)[0]
    prob = model.predict_proba(new_scaled)[0][1]

    if pred == 1:
        st.error(
            f"⚠️ High Attrition Risk ({prob:.2%})"
        )

    else:
        st.success(
            f"✅ Low Attrition Risk ({1-prob:.2%})"
        )

# ===========================
# Dataset
# ===========================
st.subheader("📁 Dataset")

st.dataframe(
    df,
    use_container_width=True
)
