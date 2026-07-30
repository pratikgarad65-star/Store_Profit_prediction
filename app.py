import streamlit as st
import numpy as np
import pandas as pd

# 1. Main Heading
st.title("Store Profit Prediction")
st.write("Select the options below to predict the estimated annual profit for your store.")

st.markdown("---")

# 2. MCQ / Selection Inputs
st.subheader("Store Information")

# MCQ 1: Store Location / Region
state = st.selectbox(
    "1. Select the State / Location:",
    options=["New York", "California", "Florida", "Texas", "Illinois"]
)

# MCQ 2: Store Type
store_type = st.radio(
    "2. Select Store Type:",
    options=["Supermarket", "Express Outlet", "Flagship Store", "Departmental Store"]
)

# MCQ 3: Location Tier
location_tier = st.radio(
    "3. Select City Tier:",
    options=["Tier 1 (Metro)", "Tier 2 (Urban)", "Tier 3 (Suburban)"]
)

# MCQ 4: R&D / Innovation Budget Range
rd_budget = st.selectbox(
    "4. Select R&D & Tech Investment Range:",
    options=[
        "Low ($0 - $30,000)",
        "Medium ($30,001 - $80,000)",
        "High ($80,001 - $150,000)",
        "Very High ($150,000+)"
    ]
)

# MCQ 5: Marketing Spend Range
marketing_spend = st.selectbox(
    "5. Select Marketing Spend Range:",
    options=[
        "Low ($0 - $50,000)",
        "Moderate ($50,001 - $150,000)",
        "High ($150,001 - $300,000)",
        "Aggressive ($300,000+)"
    ]
)

# MCQ 6: Admin / Operational Expenses
admin_spend = st.selectbox(
    "6. Select Operational / Admin Expenses:",
    options=[
        "Budget ($50,000 - $80,000)",
        "Standard ($80,001 - $120,000)",
        "Premium ($120,001+)"
    ]
)

st.markdown("---")

# 3. Prediction Action
if st.button("Predict Profit", type="primary"):
    # Convert chosen options to dummy numerical values for illustration 
    # (Replace this logic with your trained model: model.predict(...))
    
    base_profit = 50000
    
    # Simple logic boost for demonstration
    if "High" in rd_budget or "Very High" in rd_budget:
        base_profit += 40000
    if "High" in marketing_spend or "Aggressive" in marketing_spend:
        base_profit += 35000
    if "Tier 1" in location_tier:
        base_profit += 20000

    st.success(f"**Estimated Predicted Profit:** ${base_profit:,.2f}")
