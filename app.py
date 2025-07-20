import os
import io

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF

# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------
sns.set(style="whitegrid")
plt.style.use("ggplot")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def generate_summary_text(cluster_summary: pd.DataFrame) -> str:
    """Return the multi‑line textual summary with recommendations."""
    summary_text = ""
    for i, row in cluster_summary.iterrows():
        summary_text += f"Segment {i}:\n"
        summary_text += f"- Average Age: {row['Age']} years\n"
        summary_text += f"- Average Income: ${row['Annual Income (k$)']}k\n"
        summary_text += f"- Spending Score: {row['Spending Score (1-100)']}\n"

        if row['Spending Score (1-100)'] > 70 and row['Annual Income (k$)'] > 70:
            summary_text += "Recommendation: 💎 High‑value spenders. Provide premium service, exclusive offers.\n"
        elif row['Spending Score (1-100)'] > 70:
            summary_text += "Recommendation: 🛍️ Trendy shoppers. Focus on lifestyle brands and quick checkout.\n"
        elif row['Annual Income (k$)'] > 70:
            summary_text += "Recommendation: 💼 Wealthy but cautious. Trigger spending with smart promotions.\n"
        else:
            summary_text += "Recommendation: 🧺 Budget‑friendly. Promote savings, discounts, essentials.\n"
        summary_text += "\n"
    return summary_text


def build_pdf(summary_text: str) -> bytes:
    """Generate a PDF and return as raw bytes."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Smart Mall Customer Segmentation Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    # Write each line; multi_cell handles wrapping
    for line in summary_text.strip().split('\n'):
        line_clean = line.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 8, line_clean)

    return pdf.output(dest="S").encode("latin-1")


def get_recommendations(persona: str, age: int):
    """Return a list of store recommendations for the SmartMall assistant."""
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

    return []


def age_message(age: int) -> str:
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

# -----------------------------------------------------------------------------
# Streamlit app
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Smart Mall Segmentation", layout="wide")
st.title("🛍️ Smart Mall Customer Segmentation & Assistant")

# Sidebar – Data upload & parameters
st.sidebar.header("📂 Data & Parameters")
uploaded_file = st.sidebar.file_uploader("Upload your Mall Customers CSV", type=["csv"])

# Optional sample dataset fallback
sample_path = "Mall_Customers.csv"
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
elif os.path.exists(sample_path):
    df = pd.read_csv(sample_path)
    st.sidebar.info("Using bundled sample 'Mall_Customers.csv'.")
else:
    st.warning("Please upload a data file to continue.")
    st.stop()

# Standardise column names if necessary
expected_cols = ['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']
if list(df.columns) != expected_cols:
    df.columns = expected_cols  # assumes same number/order

# Main tabs
info_tab, viz_tab, cluster_tab, assistant_tab = st.tabs(["📊 Exploratory", "📈 Visualisations", "🔍 Clustering", "🤖 SmartMall Assistant"])

# -----------------------------------------------------------------------------
# Exploratory tab
# -----------------------------------------------------------------------------
with info_tab:
    st.subheader("Dataset Overview")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    st.dataframe(df.head())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Descriptive Statistics")
    st.write(df.describe())

# -----------------------------------------------------------------------------
# Visualisations tab
# -----------------------------------------------------------------------------
with viz_tab:
    st.subheader("Univariate & Bivariate Plots")

    # Gender Countplot
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Gender', data=df, palette='Set2', ax=ax1)
    ax1.set_title('Gender Distribution')
    st.pyplot(fig1)

    # Age Histogram
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.histplot(df['Age'], bins=15, kde=True, color='skyblue', ax=ax2)
    ax2.set_title('Customer Age Distribution')
    st.pyplot(fig2)

    # Income vs Spending by Gender
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender', palette='coolwarm', ax=ax3)
    ax3.set_title('Income vs Spending Score by Gender')
    st.pyplot(fig3)

# -----------------------------------------------------------------------------
# Clustering tab
# -----------------------------------------------------------------------------
with cluster_tab:
    st.subheader("K‑Means Segmentation")

    # Feature selection
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Elbow Method plot (cached for speed)
    @st.cache_data(show_spinner=False)
    def elbow_plot(data):
        inertia = []
        for k in range(1, 11):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(data)
            inertia.append(km.inertia_)
        return inertia

    inertia = elbow_plot(scaled_features)
    fig_elbow, ax_e = plt.subplots(figsize=(8, 4))
    ax_e.plot(range(1, 11), inertia, marker='o')
    ax_e.set_xlabel('Number of Clusters')
    ax_e.set_ylabel('Inertia')
    ax_e.set_title('Elbow Method for Optimal k')
    st.pyplot(fig_elbow)

    k_default = 5
    k = st.number_input("Choose the number of clusters (k)", min_value=2, max_value=10, value=k_default)

    # Run KMeans
    km_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = km_model.fit_predict(scaled_features)

    # 2D Cluster visualisation
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set2', s=100, ax=ax4)
    ax4.set_title('Customer Segments based on Income & Spending')
    st.pyplot(fig4)

    # Cluster summary table
    cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
    cluster_summary['Count'] = df['Cluster'].value_counts().sort_index()
    st.write("### Segment Summary")
    st.dataframe(cluster_summary)

    summary_text = generate_summary_text(cluster_summary)
    st.text(summary_text)

    # Additional plots
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Cluster', palette='Set2', ax=ax5)
    ax5.set_title('Customer Count per Segment')
    st.pyplot(fig5)

    for col in ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']:
        fig_box, ax_box = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=df, x='Cluster', y=col, palette='Pastel2', ax=ax_box)
        ax_box.set_title(f'{col} distribution across Segments')
        st.pyplot(fig_box)

    # PDF download
    pdf_bytes = build_pdf(summary_text)
    st.download_button("📄 Download Summary PDF", data=pdf_bytes, file_name="smart_mall_summary.pdf", mime="application/pdf")

# -----------------------------------------------------------------------------
# SmartMall Assistant tab
# -----------------------------------------------------------------------------
with assistant_tab:
    st.subheader("Personalised Store Suggestions")

    age = st.number_input("Enter your age", min_value=1, max_value=120, value=25, step=1)
    personas = ["Young Spender", "Luxury Lover", "Budget Shopper", "Balanced Buyer"]
    persona_choice = st.selectbox("Choose your shopping style/persona", options=personas)

    if st.button("Get Recommendations"):
        recs = get_recommendations(persona_choice, age)
        msg = age_message(age)
        st.markdown(f"**{msg}**")
        st.markdown(f"### Recommended Stores for a **{persona_choice}** aged {age}")
        for store in recs:
            st.write(f"• {store}")
