# ============================================
# Imports
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

import shap

# ============================================
# Load Data
# ============================================
df = pd.read_csv("data.csv")

print("\n=== Data Check ===")
print("Shape:", df.shape)

# ============================================
# Encoding
# ============================================
label_cols = [
    'Attrition','BusinessTravel','Department','EducationField',
    'Gender','JobRole','MaritalStatus','Over18','OverTime'
]

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# ============================================
# Feature Engineering (SAME LOGIC)
# ============================================
df['EmployeeEngagement'] = (df['JobInvolvement'] + df['WorkLifeBalance']) / 2

df['EmployeeSatisfaction'] = (
    df['JobSatisfaction'] +
    df['EnvironmentSatisfaction'] +
    df['RelationshipSatisfaction']
) / 3

df['GrowthRisk'] = df['YearsSinceLastPromotion'] + df['YearsInCurrentRole']
df['StressScore'] = df['OverTime'] + (5 - df['WorkLifeBalance'])

# ============================================
# ManagerID
# ============================================
df['ManagerID'] = (
    df['Department'].astype(str) + "_" +
    df['JobRole'].astype(str) + "_" +
    (df['JobLevel'] // 2).astype(str)
)

# ============================================
# Manager Info (KEEP)
# ============================================
manager_ids = df['ManagerID'].unique()

manager_names = [
    "Ahmed Hassan","Mohamed Ali","Mahmoud Samir","Omar Khaled",
    "Youssef Tarek","Mostafa Adel","Ibrahim Nasser",
    "Sara Ahmed","Mona Hany","Nour Magdy","Aya Khaled"
]

manager_df = pd.DataFrame({
    "ManagerID": manager_ids,
    "ManagerName": [random.choice(manager_names) for _ in range(len(manager_ids))],
    "ManagerCode": [f"M{100+i}" for i in range(len(manager_ids))]
})

# ============================================
# Features (Employee Level)
# ============================================
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

# ============================================
# Scaling (IMPORTANT for Logistic)
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================
# Train/Test
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# Logistic Model
# ============================================
model = LogisticRegression(max_iter=500, solver='liblinear', random_state=42)
model.fit(X_train, y_train)

# ============================================
# Evaluation
# ============================================
y_pred = model.predict(X_test)

print("\n=== Employee Attrition Model (Logistic) ===")
print(classification_report(y_test, y_pred))

y_prob = model.predict_proba(X_test)[:,1]
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# ============================================
# Predict ALL Employees
# ============================================
df['AttritionRisk'] = model.predict_proba(X_scaled)[:,1]

# ============================================
# Leader Aggregation
# ============================================
leader_df = df.groupby('ManagerID').agg({
    'AttritionRisk':'mean',
    'EmployeeEngagement':'mean',
    'EmployeeSatisfaction':'mean',
    'StressScore':'mean'
}).reset_index()

leader_df = leader_df.merge(manager_df, on="ManagerID", how="left")

# ============================================
# Text Analytics (KEEP 🔥)
# ============================================
df['Empployee Comments'] = df['Empployee Comments'].fillna("")

comments_per_manager = df.groupby('ManagerID')['Empployee Comments'] \
    .apply(lambda x: " ".join(x.astype(str))) \
    .reset_index()

tfidf = TfidfVectorizer(max_features=100)
X_text = tfidf.fit_transform(comments_per_manager['Empployee Comments']).toarray()

text_features = tfidf.get_feature_names_out()

text_df = pd.DataFrame(X_text, columns=text_features)
text_df['ManagerID'] = comments_per_manager['ManagerID']

leader_df = leader_df.merge(text_df, on='ManagerID', how='left')
leader_df[text_features] = leader_df[text_features].fillna(0)

# ============================================
# Leader Score (SAME LOGIC)
# ============================================
leader_df['LeaderScore'] = (
    (1 - leader_df['AttritionRisk']) * 0.4 +
    leader_df['EmployeeEngagement'] * 0.2 +
    leader_df['EmployeeSatisfaction'] * 0.2 +
    (1 - leader_df['StressScore']/5) * 0.2
)

# ============================================
# Classification (Percentile)
# ============================================
threshold = leader_df['LeaderScore'].quantile(0.6)

leader_df['LeaderQuality'] = np.where(
    leader_df['LeaderScore'] >= threshold, 'Good', 'Not Good'
)

good_count = (leader_df['LeaderQuality'] == 'Good').sum()
not_good_count = (leader_df['LeaderQuality'] == 'Not Good').sum()

total = len(leader_df)

print(f"\nTotal Good Leaders: {good_count} ({good_count/total:.2%})")
print(f"Total Not Good Leaders: {not_good_count} ({not_good_count/total:.2%})")

# ============================================
# Ranking
# ============================================
leader_df = leader_df.sort_values(by='LeaderScore', ascending=False)

print("\n=== Leader Ranking ===")
print(leader_df.head(10))

# ============================================
# Visualization
# ============================================
plt.figure(figsize=(14,6))

plt.bar(
    leader_df['ManagerID'],
    leader_df['LeaderScore'],
    color=['green' if x=='Good' else 'red' for x in leader_df['LeaderQuality']]
)

plt.xticks(rotation=45)
plt.title("Leader Ranking (Logistic + Text)")
plt.tight_layout()
plt.show()

# ============================================
# SHAP (Explain Logistic)
# ============================================
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# ============================================
# Insights
# ============================================
print("\n=== Insights ===")

top = leader_df.iloc[0]
worst = leader_df.iloc[-1]

print("\nTop Leader:", top['ManagerName'], f"(ID: {top['ManagerCode']})")
print(f"Attrition Risk: {top['AttritionRisk']:.2f}")
print(f"Engagement: {top['EmployeeEngagement']:.2f}")

print("\nWorst Leader:", worst['ManagerName'], f"(ID: {worst['ManagerCode']})")
print(f"Attrition Risk: {worst['AttritionRisk']:.2f}")
print(f"Stress Score: {worst['StressScore']:.2f}")

# ============================================
# Save
# ============================================
leader_df.to_csv("leader_analysis_logistic.csv", index=False)

print("\nDONE")