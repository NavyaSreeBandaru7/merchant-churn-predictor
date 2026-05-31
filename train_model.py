import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

print("🔧 Generating synthetic Shopify merchant data...")

# Generate 1,000 synthetic merchants
n_merchants = 1000

# Features: what Shopify would track
data = {
    'store_age_days': np.random.randint(1, 1095, n_merchants),  # 0-3 years
    'monthly_revenue': np.random.exponential(scale=5000, size=n_merchants),  # Right-skewed like real e-commerce
    'days_since_last_sale': np.random.randint(0, 180, n_merchants),
    'active_products': np.random.randint(1, 200, n_merchants),
    'total_customers': np.random.randint(1, 5000, n_merchants),
    'refund_rate': np.random.uniform(0, 0.3, n_merchants),  # 0-30% refund rate
    'avg_order_value': np.random.uniform(20, 500, n_merchants),
    'monthly_visitors': np.random.randint(10, 50000, n_merchants),
    'subscription_plan': np.random.choice([0, 1, 2], n_merchants),  # 0=basic, 1=shopify, 2=advanced
    'marketing_spend_ratio': np.random.uniform(0, 0.5, n_merchants),  # % of revenue spent on marketing
}

df = pd.DataFrame(data)

# Create churn labels based on realistic patterns
# Merchants churn when: inactive for long, low revenue, high refund rate, young store
churn_probability = (
    (df['days_since_last_sale'] / 180) * 0.4 +  # Inactivity = 40% of decision
    (1 / (1 + df['monthly_revenue'] / 5000)) * 0.3 +  # Low revenue = 30% of decision
    (df['refund_rate']) * 0.15 +  # High refunds = 15% of decision
    (1 / (1 + df['store_age_days'] / 365)) * 0.15  # Young stores = 15% of decision
)

# Add noise so it's not perfectly deterministic
churn_probability += np.random.normal(0, 0.1, n_merchants)
churn_probability = np.clip(churn_probability, 0, 1)

df['churned'] = (churn_probability > 0.5).astype(int)

print(f"✅ Generated {len(df)} merchants")
print(f"   Churn rate: {df['churned'].mean():.1%}")
print(f"\nData sample:")
print(df.head(10))

# Train/test split
X = df.drop('churned', axis=1)
y = df['churned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n🎯 Training Random Forest model...")

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Model trained!")
print(f"\n📊 Performance on test set ({len(X_test)} merchants):")
print(f"   Accuracy:  {accuracy:.1%}")
print(f"   Precision: {precision:.1%} (of merchants we flag as at-risk, this many actually churn)")
print(f"   Recall:    {recall:.1%} (of merchants who churn, we catch this many)")
print(f"   F1 Score:  {f1:.1%}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n🎯 Top predictors of churn:")
for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']:.<30} {row['importance']:.1%}")

# Save model
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/churn_model.pkl')
df.to_csv('model/merchant_data.csv', index=False)
feature_importance.to_csv('model/feature_importance.csv', index=False)

print(f"\n💾 Saved:")
print(f"   ✓ model/churn_model.pkl")
print(f"   ✓ model/merchant_data.csv")
print(f"   ✓ model/feature_importance.csv")
print(f"\n🚀 Ready to run: streamlit run app.py")
