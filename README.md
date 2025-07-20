# 🛍️ SmartMall Assistant — A Privacy-Aware Recommender System Using Clustering

## 🎯 Overview

**SmartMall Assistant** is a machine learning-based chatbot that recommends store visits to mall shoppers — *without asking for income or spending data*. It uses **unsupervised learning** to classify shoppers into personas and makes smart, ethical recommendations based on cluster behavior.

---

## 📊 Step 1: Data Analysis & Feature Prep

The dataset includes:
- `Age`
- `Annual Income (k$)`
- `Spending Score (1–100)`

We normalize features using **StandardScaler** to ensure unbiased clustering. Exploratory visualizations like:
- Age vs. Spending
- Income vs. Age  
reveal natural consumer patterns.

---

## 🔍 Step 2: Clustering via K-Means

We apply **K-Means** to identify patterns without labels. Using the **Elbow Method**, we pick `k=5` for meaningful segmentation. Each cluster shows unique behavioral traits (e.g., luxury buyers, budget seekers, cautious spenders).

---

## 🧠 Step 3: Creating Shopping Personas

We define 4 abstract shopper types:
- **Young Spender**
- **Luxury Lover**
- **Budget Shopper**
- **Balanced Buyer**

These are mapped to clusters based on their average behavior — *without revealing actual income or spending data*.

---

## 🤖 Step 4: Chatbot Assistant (Privacy-Respecting)

The chatbot only asks:
- 🎂 Your Age
- 🛍️ Your Shopping Style (pick from 4)

Behind the scenes:
- Your inputs are mapped to the best-matching cluster.
- The chatbot recommends stores **based on that cluster’s behavior**.

🧠 *You can’t trick it — if you say you’re 55 and a Young Spender, the chatbot uses the cluster matching your real age.*

✅ Key Benefit:
- No sensitive data collected.
- Personalized output.
- Preserved user trust.

---

## 📈 Step 5: Segment Analysis

Example segments:

### 🧺 Segment 0 – Budget-Conscious Shoppers
- **Avg Age**: 46.25  
- **Avg Income**: \$26.75k  
- **Spending**: 18.35  
- **Recommendation**: Discounts, essentials, saving-focused.

### 💎 Segment 2 – Luxury Lovers
- **Avg Age**: 32.88  
- **Avg Income**: \$86.1k  
- **Spending**: 81.53  
- **Recommendation**: Premium brands, exclusives.

### 💼 Segment 3 – Wealthy but Cautious
- **Avg Age**: 39.87  
- **Avg Income**: \$86.1k  
- **Spending**: 19.36  
- **Recommendation**: Smart offers, subtle nudges.

These help **mall managers** tailor store promotions and layouts.

---

## 📄 Step 6: PDF Report for Strategy

An auto-generated **PDF summary** compiles:
- Cluster insights
- Graphs
- Marketing suggestions

🧾 Useful for: mall management, analytics, marketing teams.

---

## 🔐 Privacy by Design

We never ask for income or spending scores. Instead:
- Clustering infers behavior
- User enters only **age** and **shopping style**
- No PII collected

✅ *A real-world ethical AI application that balances privacy and personalization.*

---

## 🧠 Real-World Benefits

- 🛒 **Shoppers**: Feel guided, not probed.
- 🏬 **Malls**: Know who’s visiting, adjust offers/stores.
- 📊 **Analysts**: Gain rich segmentation insights.
- 💻 **Devs**: Reusable pipeline for clustering + interaction.

---

## 💻 Tech Stack

- Python (Google Colab)
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-Learn (KMeans, Preprocessing)
- FPDF (PDF Report Generation)

---

## 🚀 How to Run

1. Open [`SmartMallAssistant.ipynb`](./SmartMallAssistant.ipynb) in **Google Colab**.
2. Run cells sequentially:
   - Load and scale data
   - Train KMeans
   - Visualize clusters
   - Run chatbot interface
   - Generate PDF business report
3. Done! You can now explore user segments and simulate recommendations.

---

## 📁 Files Included

| File                    | Description                                  |
|-------------------------|----------------------------------------------|
| `SmartMallAssistant.ipynb` | Full notebook (Colab-ready)               |
| `cluster_summary.pdf`   | Auto-generated business insight report       |
| `README.md`             | This file                                    |

---

## 🧪 Future Enhancements

- 🗣️ NLP chatbot (natural conversation)
- 🌐 Deploy via **Streamlit** or **Flask**
- 🔁 Feedback loops to improve clustering
- 🎯 Predict store visits with Reinforcement Learning

---

Let SmartMall Assistant guide your mall — **customer by customer, cluster by cluster** — while keeping privacy at the core.
