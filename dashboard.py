import streamlit as st
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='font-size: 35px;'>🌿💊 WellSpring Pharmacy: Customer Call Insight Dashboard </h1>",
    unsafe_allow_html=True
)
st.markdown("Analyze trends from customer support call summaries.")

# --- Load data ---
@st.cache_data
def load_json_data(folder="./reports/json"):
    data = []
    for file in os.listdir(folder):
        if file.endswith(".json"):
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                data.append(json.load(f))
    return pd.DataFrame(data)

df = load_json_data()
df = df.drop(columns='model')

# Sidebar filters
with st.expander("🔍 Filters", expanded=True):
    selected_agents = st.multiselect("Choose Representatives", df['representative_id'].unique(), default=df['representative_id'].unique())
    selected_intents = st.multiselect("Choose Intents", df['intent'].unique(), default=df['intent'].unique())

# Filter data
filtered_df = df[(df['representative_id'].isin(selected_agents)) & (df['intent'].isin(selected_intents))]



# Section: Agents Performance Check
st.markdown("### 📊 Call Overview")


# Layout: 2 columns
col1, col2 = st.columns(2)

# Chart 1: Intent Distribution
with col1:
    st.markdown("##### 📌Overall Intent Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(data=filtered_df, x='intent', order=filtered_df['intent'].value_counts().index, ax=ax1, palette='Set2')
    ax1.set_xlabel("")
    ax1.set_ylabel("Count")
    plt.xticks(rotation=30)
    st.pyplot(fig1)

# Chart 2: Sentiment by Intent
with col2:
    st.markdown("##### 📌Sentiment Ending by Call Intent")
    fig2, ax2 = plt.subplots()
    sns.countplot(data=filtered_df, x='intent', hue='sentiment_ending', ax=ax2, palette='Set2')
    ax2.set_xlabel("")
    ax2.set_ylabel("Number of Calls")
    plt.xticks(rotation=30)
    st.pyplot(fig2)


# Add a line to separate sections
st.markdown("<hr style='border: 1px solid #ddd; margin: 25px 0;'>", unsafe_allow_html=True)


# Section: Agents Performance Check
st.markdown("### 👩‍💼👨‍💼 Agent Performance")

col3, col4 = st.columns(2)

# Chart: Stacked Sentiment Counts per Agent
with col3:
    #st.subheader("🧱 Stacked Sentiment Counts per Representative")
    st.markdown("##### 📌Stacked Sentiment Counts per Representative")
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sentiment_counts = filtered_df.groupby(['representative_id', 'sentiment_ending']).size().unstack(fill_value=0)
    sentiment_counts[['neutral', 'positive']].plot(kind='bar', stacked=True, ax=ax3, color=['#66c2a5', '#fc8d62'])
    ax3.set_ylabel("Number of Calls")
    st.pyplot(fig3, use_container_width=True)

with col4:
    st.markdown("##### 📌Positive Call Rate per Agent")
    positive_rate = df.groupby('representative_id')['sentiment_ending'].value_counts(normalize=True).unstack().fillna(0)
    fig6, ax6 = plt.subplots(figsize=(6, 5))
    positive_rate['positive'].plot(kind='bar', color='#8da0cb')
    ax6.set_ylabel('Proportion of Calls Ending Positive')
    st.pyplot(fig6, use_container_width=True)

# Add a line to separate sections
st.markdown("<hr style='border: 1px solid #ddd; margin: 25px 0;'>", unsafe_allow_html=True)

# Section: Keyword Insights
st.markdown("### 🧠 Keyword Insights")

col5, col6 = st.columns(2)

with col5:
    st.markdown("##### 📌Word Cloud")
    all_keywords = [kw for sublist in filtered_df['keywords'] for kw in sublist]
    wordcloud = WordCloud(width=600, height=400, background_color='white').generate(" ".join(all_keywords))
    fig3, ax3 = plt.subplots()
    ax3.imshow(wordcloud, interpolation='bilinear')
    ax3.axis("off")
    st.pyplot(fig3)

with col6:
    st.markdown("##### 📌Top Keywords")
    keyword_series = pd.Series(all_keywords).value_counts().head(15)
    fig4 = plt.figure(figsize=(6, 6))
    sns.barplot(x=keyword_series.values, y=keyword_series.index, color = '#8da0cb')
    plt.xlabel("Frequency")
    st.pyplot(fig4)
