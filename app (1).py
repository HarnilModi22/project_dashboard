import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Google Play Store Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_DIR = "data"
APPS_PATH = f"{DATA_DIR}/apps.csv"
REVIEWS_PATH = f"{DATA_DIR}/user_reviews.csv"


# ----------------------------------------------------------------------------
# Data loading & cleaning
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    apps = pd.read_csv(APPS_PATH)
    reviews = pd.read_csv(REVIEWS_PATH)

    # Drop stray index column if present
    if "Unnamed: 0" in apps.columns:
        apps = apps.drop(columns=["Unnamed: 0"])

    # Drop exact duplicate app rows, keep the one with most reviews
    apps["Reviews"] = pd.to_numeric(apps["Reviews"], errors="coerce")
    apps = apps.sort_values("Reviews", ascending=False).drop_duplicates(
        subset="App", keep="first"
    )

    # Clean Installs: "10,000+" -> 10000
    apps["Installs_Clean"] = (
        apps["Installs"]
        .astype(str)
        .str.replace(r"[+,]", "", regex=True)
        .replace("Free", np.nan)
    )
    apps["Installs_Clean"] = pd.to_numeric(apps["Installs_Clean"], errors="coerce")

    # Clean Price: "$4.99" -> 4.99
    apps["Price_Clean"] = (
        apps["Price"].astype(str).str.replace("$", "", regex=False)
    )
    apps["Price_Clean"] = pd.to_numeric(apps["Price_Clean"], errors="coerce").fillna(0)

    # Clean Size: numeric MB, "Varies with device" -> NaN
    apps["Size_Clean"] = pd.to_numeric(apps["Size"], errors="coerce")

    # Rating numeric, drop impossible values (>5)
    apps["Rating"] = pd.to_numeric(apps["Rating"], errors="coerce")
    apps.loc[apps["Rating"] > 5, "Rating"] = np.nan

    # Last Updated -> datetime
    apps["Last_Updated_Clean"] = pd.to_datetime(apps["Last Updated"], errors="coerce")
    apps["Update_Year"] = apps["Last_Updated_Clean"].dt.year

    # Clean reviews merge dataset
    reviews = reviews.dropna(subset=["Translated_Review"])
    reviews = reviews[reviews["Translated_Review"].astype(str).str.lower() != "nan"]

    return apps, reviews


with st.spinner("Loading data..."):
    apps_df, reviews_df = load_data()

# ----------------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------------
st.sidebar.title("📱 Filters")

categories = sorted(apps_df["Category"].dropna().unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Category", options=categories, default=[]
)

types = sorted(apps_df["Type"].dropna().unique().tolist())
selected_types = st.sidebar.multiselect("Type", options=types, default=[])

content_ratings = sorted(apps_df["Content Rating"].dropna().unique().tolist())
selected_content = st.sidebar.multiselect(
    "Content Rating", options=content_ratings, default=[]
)

min_rating, max_rating = st.sidebar.slider(
    "Rating range", 0.0, 5.0, (0.0, 5.0), step=0.1
)

# Apply filters
df = apps_df.copy()
if selected_categories:
    df = df[df["Category"].isin(selected_categories)]
if selected_types:
    df = df[df["Type"].isin(selected_types)]
if selected_content:
    df = df[df["Content Rating"].isin(selected_content)]
df = df[
    (df["Rating"].isna()) | ((df["Rating"] >= min_rating) & (df["Rating"] <= max_rating))
]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(df):,}** of {len(apps_df):,} apps")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.title("📱 Google Play Store Dashboard")
st.caption(
    "Explore app ratings, installs, pricing, categories, and user sentiment "
    "from the Google Play Store dataset."
)

tab_overview, tab_categories, tab_installs_price, tab_sentiment, tab_explorer = st.tabs(
    ["📊 Overview", "🗂 Categories", "💰 Installs & Price", "💬 Sentiment", "🔎 App Explorer"]
)

# ----------------------------------------------------------------------------
# Overview tab
# ----------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Apps", f"{len(df):,}")
    c2.metric("Avg Rating", f"{df['Rating'].mean():.2f}" if df["Rating"].notna().any() else "N/A")
    c3.metric("Free Apps", f"{(df['Type'] == 'Free').sum():,}")
    c4.metric("Paid Apps", f"{(df['Type'] == 'Paid').sum():,}")

    st.markdown("### Rating Distribution")
    fig = px.histogram(
        df.dropna(subset=["Rating"]),
        x="Rating",
        nbins=40,
        color_discrete_sequence=["#4C78A8"],
    )
    fig.update_layout(xaxis_title="Rating", yaxis_title="Number of Apps", bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Apps by Category")
        cat_counts = df["Category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig = px.bar(
            cat_counts.head(15),
            x="Count",
            y="Category",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Free vs Paid")
        type_counts = df["Type"].value_counts().reset_index()
        type_counts.columns = ["Type", "Count"]
        fig = px.pie(type_counts, names="Type", values="Count", hole=0.5)
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Categories tab
# ----------------------------------------------------------------------------
with tab_categories:
    st.markdown("### Average Rating by Category")
    cat_rating = (
        df.dropna(subset=["Rating"])
        .groupby("Category")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        cat_rating,
        x="Rating",
        y="Category",
        orientation="h",
        color="Rating",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Total Installs by Category")
    cat_installs = (
        df.dropna(subset=["Installs_Clean"])
        .groupby("Category")["Installs_Clean"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig = px.bar(
        cat_installs,
        x="Installs_Clean",
        y="Category",
        orientation="h",
        color="Installs_Clean",
        color_continuous_scale="Blues",
        labels={"Installs_Clean": "Total Installs"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=700)
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Installs & Price tab
# ----------------------------------------------------------------------------
with tab_installs_price:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Rating vs. Number of Reviews")
        sample = df.dropna(subset=["Rating", "Reviews"])
        if len(sample) > 3000:
            sample = sample.sample(3000, random_state=42)
        fig = px.scatter(
            sample,
            x="Reviews",
            y="Rating",
            color="Type",
            log_x=True,
            hover_name="App",
            opacity=0.6,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("### Installs Distribution")
        install_counts = (
            df["Installs"].value_counts().reset_index()
        )
        install_counts.columns = ["Installs", "Count"]
        fig = px.bar(install_counts, x="Installs", y="Count", color="Count", color_continuous_scale="Teal")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top 20 Most Expensive Paid Apps")
    paid = df[df["Type"] == "Paid"].sort_values("Price_Clean", ascending=False).head(20)
    st.dataframe(
        paid[["App", "Category", "Price_Clean", "Rating", "Installs", "Reviews"]].rename(
            columns={"Price_Clean": "Price ($)"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Size vs Rating")
    sample2 = df.dropna(subset=["Size_Clean", "Rating"])
    if len(sample2) > 3000:
        sample2 = sample2.sample(3000, random_state=42)
    fig = px.scatter(
        sample2, x="Size_Clean", y="Rating", color="Type", opacity=0.5,
        labels={"Size_Clean": "Size (MB)"}, hover_name="App",
    )
    st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# Sentiment tab
# ----------------------------------------------------------------------------
with tab_sentiment:
    apps_in_view = set(df["App"].unique())
    rev = reviews_df[reviews_df["App"].isin(apps_in_view)]

    if rev.empty:
        st.info("No review data available for the current filter selection.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Sentiment Breakdown")
            sent_counts = rev["Sentiment"].value_counts().reset_index()
            sent_counts.columns = ["Sentiment", "Count"]
            color_map = {"Positive": "#2ca02c", "Negative": "#d62728", "Neutral": "#7f7f7f"}
            fig = px.pie(
                sent_counts, names="Sentiment", values="Count", hole=0.5,
                color="Sentiment", color_discrete_map=color_map,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("### Sentiment Polarity Distribution")
            fig = px.histogram(
                rev.dropna(subset=["Sentiment_Polarity"]),
                x="Sentiment_Polarity",
                nbins=40,
                color="Sentiment",
                color_discrete_map=color_map,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top Apps by Average Sentiment Polarity (min. 20 reviews)")
        top_sent = (
            rev.groupby("App")
            .agg(avg_polarity=("Sentiment_Polarity", "mean"), n_reviews=("Sentiment_Polarity", "count"))
            .query("n_reviews >= 20")
            .sort_values("avg_polarity", ascending=False)
            .head(15)
            .reset_index()
        )
        fig = px.bar(
            top_sent, x="avg_polarity", y="App", orientation="h",
            color="avg_polarity", color_continuous_scale="RdYlGn",
            labels={"avg_polarity": "Avg. Sentiment Polarity"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------------------------
# App Explorer tab
# ----------------------------------------------------------------------------
with tab_explorer:
    st.markdown("### Search & Explore Individual Apps")
    search = st.text_input("Search by app name", "")
    view_df = df.copy()
    if search:
        view_df = view_df[view_df["App"].str.contains(search, case=False, na=False)]

    st.dataframe(
        view_df[
            [
                "App", "Category", "Rating", "Reviews", "Installs",
                "Type", "Price", "Content Rating", "Genres", "Last Updated",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        height=400,
    )

    st.markdown("### App Reviews")
    app_choice = st.selectbox(
        "Select an app to view its reviews",
        options=sorted(view_df["App"].unique().tolist()) if len(view_df) else [],
    )
    if app_choice:
        app_reviews = reviews_df[reviews_df["App"] == app_choice]
        if app_reviews.empty:
            st.info("No reviews found for this app in the dataset.")
        else:
            st.dataframe(
                app_reviews[["Translated_Review", "Sentiment", "Sentiment_Polarity"]],
                use_container_width=True,
                hide_index=True,
                height=350,
            )

st.markdown("---")
st.caption("Data source: Google Play Store Apps dataset. Built with Streamlit.")
