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
