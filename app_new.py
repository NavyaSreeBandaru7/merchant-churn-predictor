import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============ AUTO-GENERATE MODEL IF NOT EXISTS ============
@st.cache_resource
def generate_and_load_model():
    """Generate model on first run, then load it"""
    model_path = 'model/churn_model.pkl'
    feature_importance_path = 'model/feature_importance.csv'
    merchant_data_path = 'model/merchant_data.csv'
    
    # Check if model already exists
    if os.path.exists(model_path) and os.path.exists(feature_importance_path):
        # Load existing model
        model = joblib.load(model_path)
        feature_importance = pd.read_csv(feature_importance_path)
        merchant_data = pd.read_csv(merchant_data_path)
        return model, feature_importance, merchant_data
    
    # Generate new model if it doesn't exist
    st.info("⏳ Generating model on first run (takes ~15 seconds)...")
    
    # Set random seed
    np.random.seed(42)
    
    # Generate synthetic data
    n_merchants = 1000
    
    data = {
        'store_age_days': np.random.randint(1, 1095, n_merchants),
        'monthly_revenue': np.random.exponential(scale=5000, size=n_merchants),
        'days_since_last_sale': np.random.randint(0, 180, n_merchants),
        'active_products': np.random.randint(1, 200, n_merchants),
        'total_customers': np.random.randint(1, 5000, n_merchants),
        'refund_rate': np.random.uniform(0, 0.3, n_merchants),
        'avg_order_value': np.random.uniform(20, 500, n_merchants),
        'monthly_visitors': np.random.randint(10, 50000, n_merchants),
        'subscription_plan': np.random.choice([0, 1, 2], n_merchants),
        'marketing_spend_ratio': np.random.uniform(0, 0.5, n_merchants),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn labels
    churn_probability = (
        (df['days_since_last_sale'] / 180) * 0.4 +
        (1 / (1 + df['monthly_revenue'] / 5000)) * 0.3 +
        (df['refund_rate']) * 0.15 +
        (1 / (1 + df['store_age_days'] / 365)) * 0.15
    )
    
    churn_probability += np.random.normal(0, 0.1, n_merchants)
    churn_probability = np.clip(churn_probability, 0, 1)
    
    df['churned'] = (churn_probability > 0.5).astype(int)
    
    # Train model
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Calculate feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Save model and data
    joblib.dump(model, model_path)
    df.to_csv(merchant_data_path, index=False)
    feature_importance.to_csv(feature_importance_path, index=False)
    
    st.success("✅ Model generated successfully!")
    
    return model, feature_importance, df

# Load model (cached, only runs once)
model, feature_importance, merchant_data = generate_and_load_model()

# ============ PAGE CONFIG ============
st.set_page_config(
    page_title="Shopify Merchant Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .high-risk {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    .low-risk {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    .medium-risk {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    h1 { color: #1f77b4; }
    h2 { color: #2ca02c; }
</style>
""", unsafe_allow_html=True)

st.title("🔮 Shopify Merchant Churn Prediction")

st.markdown("""
**Predict which merchants are at risk of churning.**

This model identifies merchants likely to abandon Shopify in the next 30-60 days based on their activity patterns, revenue trends, and store health metrics.
""")

# Sidebar: Show feature importance
with st.sidebar:
    st.header("📈 Model Info")
    st.subheader("Top Churn Predictors")
    for idx, row in feature_importance.head(5).iterrows():
        col1, col2 = st.columns([2, 1])
        col1.write(row['feature'])
        col2.metric("", f"{row['importance']:.0%}")
    
    st.divider()
    st.subheader("Dataset Stats")
    st.metric("Total Merchants", len(merchant_data))
    st.metric("Churned", f"{merchant_data['churned'].sum()} ({merchant_data['churned'].mean():.1%})")

# ============ TABS ============
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Predictions", "Sample Merchants"])

# ============ TAB 1: Single Prediction ============
with tab1:
    st.header("Predict for One Merchant")
    
    col1, col2 = st.columns(2)
    
    with col1:
        store_age_days = st.slider("Store Age (days)", 1, 1095, 365)
        monthly_revenue = st.number_input("Monthly Revenue ($)", 100, 100000, 5000)
        days_since_last_sale = st.slider("Days Since Last Sale", 0, 180, 30)
        active_products = st.slider("Active Products", 1, 200, 25)
    
    with col2:
        total_customers = st.number_input("Total Customers", 1, 10000, 500)
        refund_rate = st.slider("Refund Rate (%)", 0, 30, 5) / 100
        avg_order_value = st.number_input("Avg Order Value ($)", 20, 500, 100)
        monthly_visitors = st.number_input("Monthly Visitors", 10, 50000, 1000)
    
    col3a, col3b = st.columns(2)
    with col3a:
        subscription_plan = st.radio("Plan Type", ["Basic (0)", "Shopify (1)", "Advanced (2)"], horizontal=True)
        subscription_plan = int(subscription_plan[0])
    
    with col3b:
        marketing_spend_ratio = st.slider("Marketing Spend (% of revenue)", 0, 50, 10) / 100
    
    if st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True):
        # Prepare input
        merchant_input = pd.DataFrame({
            'store_age_days': [store_age_days],
            'monthly_revenue': [monthly_revenue],
            'days_since_last_sale': [days_since_last_sale],
            'active_products': [active_products],
            'total_customers': [total_customers],
            'refund_rate': [refund_rate],
            'avg_order_value': [avg_order_value],
            'monthly_visitors': [monthly_visitors],
            'subscription_plan': [subscription_plan],
            'marketing_spend_ratio': [marketing_spend_ratio],
        })
        
        # Predict
        churn_prob = model.predict_proba(merchant_input)[0][1]
        
        # Display result with color coding
        st.divider()
        
        risk_level = "🟢 LOW RISK" if churn_prob < 0.33 else ("🟡 MEDIUM RISK" if churn_prob < 0.67 else "🔴 HIGH RISK")
        risk_class = "low-risk" if churn_prob < 0.33 else ("medium-risk" if churn_prob < 0.67 else "high-risk")
        
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h2>{risk_level}</h2>
            <h1>{churn_prob:.0%}</h1>
            <p>Probability this merchant churns in next 60 days</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Explain why
        st.subheader("📊 What's driving this prediction?")
        
        # Get feature contributions
        explanations = []
        for feature in merchant_input.columns:
            value = merchant_input[feature].iloc[0]
            importance = feature_importance[feature_importance['feature'] == feature]['importance'].values[0]
            explanations.append((feature, value, importance))
        
        explanations.sort(key=lambda x: x[2], reverse=True)
        
        st.write("**Top factors for this merchant:**")
        for feature, value, importance in explanations[:5]:
            col1, col2, col3 = st.columns([2, 1, 1])
            col1.write(f"**{feature}**")
            col2.write(f"Value: {value:.0f}" if isinstance(value, (int, float)) and value > 10 else f"Value: {value:.2f}")
            col3.metric("Importance", f"{importance:.0%}")
        
        # Recommendation
        st.divider()
        if churn_prob > 0.67:
            st.warning(f"""
            ⚠️ **Action Required**
            
            This merchant is at high risk. Recommended actions:
            - Assign to success team for proactive outreach
            - Offer targeted retention discount or free upgrade
            - Review their recent support tickets for pain points
            - Consider custom integration support
            """)
        elif churn_prob > 0.33:
            st.info(f"""
            💡 **Monitor & Engage**
            
            This merchant shows warning signs. Consider:
            - Regular check-in email
            - Feature education (apps, tools they're not using)
            - Community engagement (Shopify forums)
            """)
        else:
            st.success(f"""
            ✅ **Healthy Merchant**
            
            This merchant looks stable. Focus on:
            - Upsell to higher plan
            - Cross-sell relevant apps
            - NPS surveys to gather feedback
            """)

# ============ TAB 2: Batch Predictions ============
with tab2:
    st.header("Bulk Prediction")
    st.write("Upload a CSV with merchant data to get predictions for many at once.")
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)
        
        # Check columns match
        required_cols = [
            'store_age_days', 'monthly_revenue', 'days_since_last_sale', 
            'active_products', 'total_customers', 'refund_rate', 
            'avg_order_value', 'monthly_visitors', 'subscription_plan', 'marketing_spend_ratio'
        ]
        
        if all(col in df_upload.columns for col in required_cols):
            predictions = model.predict_proba(df_upload[required_cols])[:, 1]
            df_upload['churn_risk'] = predictions
            df_upload['risk_level'] = df_upload['churn_risk'].apply(
                lambda x: "HIGH" if x > 0.67 else ("MEDIUM" if x > 0.33 else "LOW")
            )
            
            st.dataframe(
                df_upload[['monthly_revenue', 'days_since_last_sale', 'churn_risk', 'risk_level']],
                use_container_width=True,
                height=400
            )
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Merchants", len(df_upload))
            col2.metric("High Risk", (df_upload['risk_level'] == 'HIGH').sum())
            col3.metric("Medium Risk", (df_upload['risk_level'] == 'MEDIUM').sum())
            col4.metric("Low Risk", (df_upload['risk_level'] == 'LOW').sum())
            
            # Download predictions
            csv = df_upload.to_csv(index=False)
            st.download_button(
                label="📥 Download predictions",
                data=csv,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
        else:
            st.error(f"❌ CSV missing required columns: {required_cols}")

# ============ TAB 3: Sample Merchants ============
with tab3:
    st.header("Sample Predictions")
    st.write("See predictions for 10 random merchants from the training data.")
    
    sample_merchants = merchant_data.sample(10).reset_index(drop=True)
    
    # Get predictions
    X_sample = sample_merchants.drop('churned', axis=1)
    sample_merchants['predicted_churn_risk'] = model.predict_proba(X_sample)[:, 1]
    sample_merchants['actual_churned'] = sample_merchants['churned'].map({0: 'No', 1: 'Yes'})
    sample_merchants['predicted_risk_level'] = sample_merchants['predicted_churn_risk'].apply(
        lambda x: "HIGH" if x > 0.67 else ("MEDIUM" if x > 0.33 else "LOW")
    )
    
    display_cols = [
        'monthly_revenue', 
        'days_since_last_sale',
        'active_products',
        'predicted_churn_risk',
        'predicted_risk_level',
        'actual_churned'
    ]
    
    st.dataframe(
        sample_merchants[display_cols].style.format({
            'monthly_revenue': '${:,.0f}',
            'predicted_churn_risk': '{:.0%}'
        }),
        use_container_width=True,
        height=400
    )
    
    # Model accuracy on these samples
    sample_merchants['prediction_correct'] = (
        (sample_merchants['predicted_churn_risk'] > 0.5).astype(int) == 
        sample_merchants['churned']
    )
    accuracy = sample_merchants['prediction_correct'].mean()
    st.metric("Model Accuracy on Sample", f"{accuracy:.0%}")
