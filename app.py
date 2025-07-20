<<<<<<< HEAD
from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
import os
import io

app = Flask(__name__)

@app.route('/')
def home():
    return "<h2>Smart Mall Customer Segmentation App</h2><p><a href='/analyze'>Run Analysis</a></p>"

@app.route('/analyze')
def analyze():
    url = "https://raw.githubusercontent.com/03-SudheshnaReddy/Smart-Mall-Assistant/main/Mall_Customers.csv"
    df = pd.read_csv(url)

    # Gender Countplot
    plt.figure(figsize=(6,4))
    sns.countplot(x='Gender', data=df, palette='Set2')
    plt.title('Gender Distribution')
    plt.savefig("static/gender.png")
    plt.close()

    # Age Histogram
    plt.figure(figsize=(8,5))
    sns.histplot(df['Age'], bins=15, kde=True, color='skyblue')
    plt.title('Customer Age Distribution')
    plt.savefig("static/age.png")
    plt.close()

    # Income vs Spending
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', palette='coolwarm')
    plt.title('Income vs Spending Score by Gender')
    plt.savefig("static/income_vs_spending.png")
    plt.close()

    # Clustering
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    inertia = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.savefig("static/elbow.png")
    plt.close()

    k = 5
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100)
    plt.title('Customer Segments based on Income & Spending Score')
    plt.savefig("static/segments.png")
    plt.close()

    cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
    cluster_summary['Count'] = df['Cluster'].value_counts().sort_index()

    summary_text = ""
    for i, row in cluster_summary.iterrows():
        summary_text += f"Segment {i}:\n"
        summary_text += f"- Average Age: {row['Age']} years\n"
        summary_text += f"- Average Income: ${row['Annual Income (k$)']}k\n"
        summary_text += f"- Spending Score: {row['Spending Score (1-100)']}\n"

        if row['Spending Score (1-100)'] > 70 and row['Annual Income (k$)'] > 70:
            summary_text += "Recommendation: 💎 High-value spenders. Provide premium service, exclusive offers.\n"
        elif row['Spending Score (1-100)'] > 70:
            summary_text += "Recommendation: 🛍️ Trendy shoppers. Focus on lifestyle brands and quick checkout.\n"
        elif row['Annual Income (k$)'] > 70:
            summary_text += "Recommendation: 💼 Wealthy but cautious. Trigger spending with smart promotions.\n"
        else:
            summary_text += "Recommendation: 🧺 Budget-friendly. Promote savings, discounts, essentials.\n"
        summary_text += "\n"

    # Generate PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Smart Mall Customer Segmentation Summary", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    for line in summary_text.strip().split('\n'):
        pdf.cell(200, 8, txt=line.encode('ascii', 'ignore').decode('ascii'), ln=True)

    pdf.output("static/smart_mall_summary.pdf")

    return render_template("result.html", summary=summary_text)

@app.route('/download')
def download():
    return send_file("static/smart_mall_summary.pdf", as_attachment=True)

@app.route('/chatbot')
def smartmall_assistant():
    def get_recommendations(persona, age):
        if persona == "Young Spender":
            if age < 25:
                return ["Zara", "H&M", "Miniso", "KFC"]
            elif age < 40:
                return ["Uniqlo", "Nike", "Barista", "Bookstore"]
            else:
                return ["Elegant Casuals", "Formal Wear", "Lifestyle", "Quiet Lounge"]

        elif persona == "Luxury Lover":
            if age < 40:
                return ["Gucci", "Sephora", "Fine Dining", "Spa"]
            else:
                return ["Louis Vuitton", "Rolex", "Premium Spa", "Art Gallery"]

        elif persona == "Budget Shopper":
            if age < 40:
                return ["DMart", "Big Bazaar", "Kids Zone", "McDonald's"]
            else:
                return ["Daily Needs", "Value Pharmacy", "Discount Mart", "Café Coffee Day"]

        elif persona == "Balanced Buyer":
            if age < 40:
                return ["Lifestyle", "Home Centre", "Starbucks", "Book Store"]
            else:
                return ["FabIndia", "Reliance Trends", "Home Decor", "South Indian Restaurant"]

    def get_age_message(age):
        if age < 18:
            return "👶 You're a young explorer! Let’s find something exciting for you."
        elif age < 30:
            return "✨ You're a stylish Gen Z shopper! Trendy picks coming up."
        elif age < 45:
            return "🧔 You're a savvy millennial! We've got your vibe covered."
        elif age < 60:
            return "🎯 You're a practical professional shopper. Smart choices ahead."
        else:
            return "👵 You're a graceful classic shopper. Comfort and charm await you."

    personas = ["Young Spender", "Luxury Lover", "Budget Shopper", "Balanced Buyer"]

    response_html = "<h2>🛍️ SmartMall Assistant</h2>"
    age = 28
    persona = "Luxury Lover"
    recommendations = get_recommendations(persona, age)
    age_message = get_age_message(age)

    response_html += f"<p>{age_message}</p>"
    response_html += f"<p><b>Age:</b> {age}, <b>Persona:</b> {persona}</p>"
    response_html += "<ul>"
    for store in recommendations:
        response_html += f"<li>{store}</li>"
    response_html += "</ul>"

    return response_html

if __name__ == '__main__':
    app.run(debug=True)
=======
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
>>>>>>> 33819300a9ba061b7bfc6dc8c09c274258858a09
