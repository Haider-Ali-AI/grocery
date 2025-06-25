import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Streamlit App Title
st.set_page_config(layout="wide")
st.title("🛒 Market Basket Analysis - Apriori Algorithm")

# Upload CSV
uploaded_file = st.file_uploader("Groceries_dataset.csv", type="csv")

if uploaded_file:
    # Load the data
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Preview of the Data")
    st.dataframe(data.head())

    # Top 10 Most Sold Items
    st.subheader("🥇 Top 10 Most Sold Items")
    top_items = data['itemDescription'].value_counts().head(10)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=top_items.index, y=top_items.values, palette='viridis', ax=ax)
    ax.set_xlabel("Item")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 10 Purchased Items")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Preprocessing: Add Quantity column
    data['Quantity'] = 1

    # Create Basket Format
    basket = data.groupby(['Member_number', 'itemDescription'])['Quantity'] \
                 .sum().unstack().fillna(0)

    # Binary encoding
    basket_encoded = basket.applymap(lambda x: 1 if x > 0 else 0)

    # Sidebar Parameters
    st.sidebar.header("⚙️ Apriori Settings")
    min_support = st.sidebar.slider("Minimum Support", 0.01, 0.2, 0.06, step=0.01)
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.1, 1.0, 0.4, step=0.05)
    min_lift = st.sidebar.slider("Minimum Lift", 0.5, 5.0, 1.0, step=0.1)

    # Apply Apriori
    frequent_itemsets = apriori(basket_encoded, min_support=min_support, use_colnames=True)

    # Generate Rules
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=min_lift)
    filtered_rules = rules[(rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]

    st.subheader("📊 Association Rules")
    if not filtered_rules.empty:
        st.dataframe(
            filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
            .sort_values(by='lift', ascending=False)
        )
    else:
        st.warning("No rules found with current thresholds. Try adjusting the parameters.")
else:
    st.info("👈 Please upload the 'Groceries_dataset.csv' file to begin.")
