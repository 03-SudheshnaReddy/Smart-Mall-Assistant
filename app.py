# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Streamlit App Config
# -------------------------------
st.set_page_config(page_title="SmartMall Assistant", layout="wide")
st.title("🛍️ SmartMall Assistant — A Privacy-Aware Recommender System")

st.markdown("""
Welcome to the **SmartMall Assistant** — a privacy-first system that uses clustering and personas to recommend stores **without asking for sensitive data**.  
Just tell us your **age** and **persona**, and get personalized store suggestions instantly.
""")

# -------------------------------
# Upload Data
# -------------------------------
uploaded_file = st.file_uploader("📁 Upload Mall_Customers.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Step 1: Data Preview")
    st.dataframe(df.head())

    # -------------------------------
    # Preprocessing
    # -------------------------------
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # -------------------------------
    # Step 2: Clustering with KMeans
    # -------------------------------
    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    st.subheader("🔍 Step 2: Clustering Result")
    st.write("We identified 5 unique customer segments using **K-Means Clustering**.")

    # Cluster Summary
    summary = df.groupby('Cluster').agg({
        'Age': 'mean',
        'Annual Income (k$)': 'mean',
        'Spending Score (1-100)': 'mean',
        'Cluster': 'count'
    }).rename(columns={"Cluster": "Count"}).round(2).reset_index()

    st.dataframe(summary)

    # -------------------------------
    # Step 3: Persona Definitions
    # -------------------------------
    st.subheader("🧠 Step 3: Define Your Persona")

    age_input = st.number_input("Enter your Age", min_value=10, max_value=100, step=1)
    persona = st.selectbox("Choose your Shopper Persona", [
        "Young Spender", "Luxury Lover", "Budget Shopper", "Balanced Buyer"
    ])

    # Map persona to clusters
    persona_to_cluster = {
        "Young Spender": 4,
        "Luxury Lover": 2,
        "Budget Shopper": 1,
        "Balanced Buyer": 3
    }

    # Logic check: age-persona mismatch
    warnings = {
        "Young Spender": (age_input > 40),
        "Luxury Lover": (age_input < 20),
        "Budget Shopper": (age_input < 18),
        "Balanced Buyer": False
    }
    if warnings.get(persona, False):
        st.warning("🤔 Hmm... that persona might not match your age. Be honest for better suggestions!")

    if st.button("🎯 Recommend Me Stores"):
        cluster_id = persona_to_cluster.get(persona, 3)
        segment = summary[summary['Cluster'] == cluster_id].iloc[0]

        st.success(f"✅ Based on Cluster {cluster_id} average behavior:")

        st.markdown(f"""
        - **Avg Age:** {segment['Age']} years  
        - **Avg Income:** ${segment['Annual Income (k$)']}k  
        - **Avg Spending Score:** {segment['Spending Score (1-100)']}  
        """)

        # Recommendation Logic
        if segment['Spending Score (1-100)'] > 70:
            rec = "💎 High spender — Recommend luxury and fashion stores"
        elif segment['Spending Score (1-100)'] < 30:
            rec = "💼 Cautious shopper — Recommend budget and essentials"
        else:
            rec = "🧺 Moderate — Recommend lifestyle & value stores"
        st.markdown(f"**🛍️ Recommendation:** {rec}")

        # Example Store Lists
        store_map = {
            "Young Spender": ["Zara", "Miniso", "Food Court", "Tech Gadgets"],
            "Luxury Lover": ["Gucci", "Sephora", "Fine Dining", "Spa"],
            "Budget Shopper": ["DMart", "Reliance Trends", "Toy Store", "Value Bazaar"],
            "Balanced Buyer": ["Lifestyle", "Starbucks", "Bookstore", "Home Centre"]
        }

        st.markdown("### 🏬 Suggested Stores:")
        for store in store_map.get(persona, []):
            st.markdown(f"- {store}")

    # -------------------------------
    # Step 5: Visualization
    # -------------------------------
    st.subheader("📈 Step 5: Visual Insights")

    fig1, ax1 = plt.subplots()
    sns.scatterplot(data=df, x="Age", y="Spending Score (1-100)", hue="Cluster", palette="viridis", ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Age", hue="Cluster", palette="cool", ax=ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", palette="plasma", ax=ax3)
    st.pyplot(fig3)

    # -------------------------------
    # Step 6: Business Report (Placeholder)
    # -------------------------------
    st.subheader("📄 Step 6: Business Report")
    st.markdown("📌 A downloadable PDF with insights will be added soon!")

else:
    st.info("📂 Please upload the dataset to continue.")
