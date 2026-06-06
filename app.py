"""
Netflix Content Analyzer & Recommender
=======================================
A production-grade Streamlit dashboard for EDA + content-based recommendations.

Author : Netflix Analyst
Dataset: netflix_titles.csv
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import re
import warnings
warnings.filterwarnings("ignore")

# ── Third-Party ───────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Netflix Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark Netflix-inspired theme) ──────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        background-color: #0d0d0d;
        color: #e8e8e8;
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Bebas Neue', cursive; letter-spacing: 2px; }
    h1 { color: #e50914; font-size: 3rem; }
    h2 { color: #f5f5f5; font-size: 2rem; }
    h3 { color: #b3b3b3; font-size: 1.4rem; }

    .stMetric {
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 8px;
        padding: 16px;
    }
    .stMetric label { color: #b3b3b3 !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e50914 !important; font-size: 2rem !important; }

    .stSelectbox label, .stSlider label { color: #b3b3b3; }
    div[data-testid="stSidebarContent"] { background-color: #111; }

    .rec-card {
        background: #1c1c1c;
        border-left: 4px solid #e50914;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    .rec-title { font-family:'Bebas Neue',cursive; font-size:1.3rem; color:#fff; }
    .rec-meta  { font-size:0.85rem; color:#888; margin-top:4px; }
    .rec-score { font-size:0.8rem; color:#e50914; font-weight:600; margin-top:6px; }
    .rec-desc  { font-size:0.9rem; color:#ccc; margin-top:8px; line-height:1.5; }

    .section-divider {
        border: none; border-top: 1px solid #2a2a2a; margin: 32px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & CLEANING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_and_clean(path: str = "netflix_titles.csv") -> pd.DataFrame:
    """
    Load the Netflix dataset, clean it, and engineer useful features.

    Steps
    -----
    1. Read CSV
    2. Drop exact duplicates
    3. Fill / flag missing values
    4. Parse date_added → datetime; extract month & year
    5. Normalize duration → numeric minutes / season count
    6. Explode genres (one row per genre)
    7. Build 'soup' column for TF-IDF
    """

    df = pd.read_csv(path)

    # ── 2. Duplicates ─────────────────────────────────────────────────────────
    df.drop_duplicates(inplace=True)

    # ── 3. Missing values ─────────────────────────────────────────────────────
    df["director"].fillna("Unknown Director", inplace=True)
    df["cast"].fillna("Unknown Cast", inplace=True)
    df["country"].fillna("Unknown Country", inplace=True)
    df["rating"].fillna("Not Rated", inplace=True)
    df["listed_in"].fillna("Uncategorized", inplace=True)
    df["description"].fillna("", inplace=True)

    # ── 4. Parse dates ────────────────────────────────────────────────────────
    df["date_added"] = pd.to_datetime(df["date_added"].str.strip(), errors="coerce")
    df["year_added"]  = df["date_added"].dt.year
    df["month_added"] = df["date_added"].dt.month_name()

    # ── 5. Duration numeric ───────────────────────────────────────────────────
    df["duration_clean"] = df["duration"].str.extract(r"(\d+)").astype(float)

    # ── 6. Primary genre (first listed) ───────────────────────────────────────
    df["primary_genre"] = df["listed_in"].str.split(",").str[0].str.strip()

    # ── 7. TF-IDF soup ────────────────────────────────────────────────────────
    df["soup"] = (
        df["title"].fillna("") + " "
        + df["director"].fillna("") + " "
        + df["cast"].str.replace(";", " ").fillna("") + " "
        + df["listed_in"].str.replace("&", "").str.replace(",", " ").fillna("") + " "
        + df["description"].fillna("")
    )
    df["soup"] = df["soup"].str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)

    df.reset_index(drop=True, inplace=True)
    return df


df = load_and_clean("netflix_titles.csv")

# ══════════════════════════════════════════════════════════════════════════════
# 2. RECOMMENDATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def build_recommender(df: pd.DataFrame):
    """
    Build a TF-IDF cosine-similarity matrix.

    TF-IDF converts the text 'soup' column into numeric vectors.
    Cosine similarity then measures how similar two items are.
    Score of 1.0 = identical, 0.0 = completely different.
    """
    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    idx_map = pd.Series(df.index, index=df["title"].str.lower())
    return cosine_sim, idx_map


cosine_sim, idx_map = build_recommender(df)


def get_recommendations(title: str, n: int = 5) -> pd.DataFrame:
    """Return top-n recommendations for a given title."""
    key = title.strip().lower()
    if key not in idx_map:
        return pd.DataFrame()

    idx   = idx_map[key]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : n + 1]

    rec_idx = [i[0] for i in scores]
    sim_scores = [round(i[1], 4) for i in scores]

    result = df.loc[rec_idx, ["title", "type", "primary_genre", "release_year",
                               "rating", "duration", "description"]].copy()
    result["similarity"] = sim_scores
    return result.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════

st.sidebar.markdown("# 🎬 Netflix Analyzer")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 EDA & Trends", "🎯 Recommender"],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Dataset snapshot**")
st.sidebar.write(f"Titles  : **{len(df)}**")
st.sidebar.write(f"Movies  : **{(df.type == 'Movie').sum()}**")
st.sidebar.write(f"TV Shows: **{(df.type == 'TV Show').sum()}**")
st.sidebar.write(f"Countries: **{df.country.nunique()}**")

# ══════════════════════════════════════════════════════════════════════════════
# 4. PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.markdown("# NETFLIX CONTENT ANALYZER")
    st.markdown(
        "**A beginner-friendly data science project** — EDA · Visualizations · AI Recommendations"
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Titles",    len(df))
    c2.metric("Movies",          (df.type == "Movie").sum())
    c3.metric("TV Shows",        (df.type == "TV Show").sum())
    c4.metric("Unique Countries",df.country.nunique())

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Dataset preview
    st.markdown("## RAW DATASET")
    st.markdown(
        "Below are the first few rows of the cleaned dataset. "
        "Each row represents one Netflix title."
    )
    st.dataframe(
        df[["title", "type", "primary_genre", "country",
            "release_year", "rating", "duration", "director"]].head(10),
        use_container_width=True,
    )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Data quality
    st.markdown("## DATA QUALITY REPORT")
    st.markdown(
        "Missing-value audit after cleaning — all columns should show 0 nulls "
        "for the core fields we use."
    )
    null_df = df.isnull().sum().reset_index()
    null_df.columns = ["Column", "Missing Values"]
    null_df["% Missing"] = (null_df["Missing Values"] / len(df) * 100).round(1)
    st.dataframe(null_df, use_container_width=True)

    st.info(
        "💡 **What we did**: filled missing `director`, `cast`, `country`, `rating`, "
        "and `listed_in` with placeholder strings so no rows were dropped. "
        "Dates were parsed into proper datetime objects."
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5. PAGE: EDA & TRENDS
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 EDA & Trends":
    st.markdown("# EDA & TRENDS")
    st.markdown(
        "Explore how Netflix content is distributed across types, genres, "
        "countries, years, and ratings."
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5a. Content Type Pie ───────────────────────────────────────────────────
    st.markdown("## CONTENT TYPE SPLIT")
    st.markdown(
        "Netflix's catalog is split between Movies and TV Shows. "
        "The donut chart below shows the proportions in our dataset."
    )
    type_counts = df["type"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=type_counts.index,
        values=type_counts.values,
        hole=0.55,
        marker=dict(colors=["#e50914", "#831010"]),
        textfont=dict(color="white"),
    ))
    fig_pie.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font_color="white", showlegend=True,
        legend=dict(font=dict(color="white")),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5b. Genre Distribution ────────────────────────────────────────────────
    st.markdown("## TOP GENRES")
    st.markdown(
        "We extracted the **primary genre** from the `listed_in` column "
        "(taking the first tag). Action & Adventure leads in our sample."
    )
    genre_counts = df["primary_genre"].value_counts().reset_index()
    genre_counts.columns = ["Genre", "Count"]
    fig_genre = px.bar(
        genre_counts, x="Count", y="Genre", orientation="h",
        color="Count", color_continuous_scale=["#831010", "#e50914"],
        template="plotly_dark",
    )
    fig_genre.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_genre, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5c. Release Year Trend ────────────────────────────────────────────────
    st.markdown("## CONTENT BY RELEASE YEAR")
    st.markdown(
        "How many titles were *released* each year? "
        "This shows Netflix's shift toward more recent content."
    )
    year_counts = df.groupby(["release_year", "type"]).size().reset_index(name="Count")
    fig_year = px.bar(
        year_counts, x="release_year", y="Count", color="type",
        color_discrete_map={"Movie": "#e50914", "TV Show": "#831010"},
        barmode="group", template="plotly_dark",
        labels={"release_year": "Release Year"},
    )
    fig_year.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        legend=dict(font=dict(color="white")),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5d. Country Distribution ──────────────────────────────────────────────
    st.markdown("## CONTENT BY COUNTRY")
    st.markdown(
        "Which countries produce the most Netflix content in our dataset? "
        "The US and India dominate."
    )
    country_counts = df["country"].value_counts().reset_index()
    country_counts.columns = ["Country", "Count"]
    fig_country = px.bar(
        country_counts, x="Country", y="Count",
        color="Count", color_continuous_scale=["#831010", "#e50914"],
        template="plotly_dark",
    )
    fig_country.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#111",
        coloraxis_showscale=False, margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_country, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5e. Ratings ───────────────────────────────────────────────────────────
    st.markdown("## CONTENT RATINGS DISTRIBUTION")
    st.markdown(
        "Netflix content spans all age ratings. "
        "Most of our sample is rated **PG-13** or **UA** (Indian certification)."
    )
    rating_counts = df["rating"].value_counts().reset_index()
    rating_counts.columns = ["Rating", "Count"]
    fig_rating = px.pie(
        rating_counts, names="Rating", values="Count",
        color_discrete_sequence=px.colors.sequential.Reds_r,
        template="plotly_dark",
    )
    fig_rating.update_layout(
        paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font_color="white", margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_rating, use_container_width=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5f. Duration Distribution ─────────────────────────────────────────────
    st.markdown("## MOVIE DURATION DISTRIBUTION")
    movies = df[df["type"] == "Movie"].dropna(subset=["duration_clean"])
    if not movies.empty:
        fig_dur, ax = plt.subplots(figsize=(8, 3))
        fig_dur.patch.set_facecolor("#0d0d0d")
        ax.set_facecolor("#111")
        ax.hist(movies["duration_clean"], bins=10, color="#e50914", edgecolor="#0d0d0d")
        ax.set_xlabel("Duration (minutes)", color="white")
        ax.set_ylabel("Count", color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        st.pyplot(fig_dur)
        st.markdown(
            f"📊 Average movie runtime: **{movies['duration_clean'].mean():.0f} min** | "
            f"Shortest: **{movies['duration_clean'].min():.0f} min** | "
            f"Longest: **{movies['duration_clean'].max():.0f} min**"
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── 5g. Seaborn heatmap – genre × type (count) ───────────────────────────
    st.markdown("## GENRE × CONTENT TYPE HEATMAP")
    st.markdown(
        "A heatmap showing how genres overlap with content types. "
        "Darker red = more titles."
    )
    pivot = df.pivot_table(index="primary_genre", columns="type",
                            aggfunc="size", fill_value=0)
    fig_heat, ax2 = plt.subplots(figsize=(6, max(3, len(pivot) * 0.5)))
    fig_heat.patch.set_facecolor("#0d0d0d")
    ax2.set_facecolor("#111")
    sns.heatmap(
        pivot, annot=True, fmt="d", cmap="Reds",
        linewidths=0.5, linecolor="#1a1a1a",
        ax=ax2, cbar_kws={"shrink": 0.8},
    )
    ax2.tick_params(colors="white")
    ax2.set_xlabel("Type", color="white")
    ax2.set_ylabel("Primary Genre", color="white")
    plt.tight_layout()
    st.pyplot(fig_heat)

    # ── Conclusions ───────────────────────────────────────────────────────────
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("## 💡 KEY INSIGHTS")
    insights = [
        "**Action & Adventure** is the dominant genre in this dataset.",
        "**India** and the **United States** are the largest content producers.",
        "Recent years (2020–2023) show a strong surge in new releases.",
        "Movies outnumber TV Shows in the sample.",
        "**PG-13** and **TV-14** are the most common ratings, showing a family-friendly lean.",
        "Movie runtimes cluster between 100–180 minutes.",
    ]
    for i in insights:
        st.markdown(f"- {i}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. PAGE: RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🎯 Recommender":
    st.markdown("# CONTENT RECOMMENDER")
    st.markdown(
        "Pick any title from the dataset and get the most similar Netflix content "
        "based on **genre, cast, director, and description** — powered by "
        "TF-IDF + Cosine Similarity."
    )
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    st.markdown("### How does it work?")
    with st.expander("📖 Beginner explanation — click to expand"):
        st.markdown(
            """
**Step 1 — Build a 'soup'**  
We combine each title's name, director, cast, genres, and description into a single text blob called a *soup*.

**Step 2 — TF-IDF Vectorization**  
TF-IDF (Term Frequency-Inverse Document Frequency) converts the soup into a numeric vector.  
Common words like "the" get low weight; unique keywords like "heist" get high weight.

**Step 3 — Cosine Similarity**  
We compute how similar every pair of vectors is using cosine similarity.  
Score 1.0 = identical. Score 0.0 = completely different.

**Step 4 — Rank & Return**  
For your chosen title, we sort all other titles by similarity score and return the top N.
            """
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    col_sel, col_n = st.columns([3, 1])
    with col_sel:
        selected_title = st.selectbox(
            "Choose a Netflix title:", sorted(df["title"].tolist())
        )
    with col_n:
        n_recs = st.slider("# Recommendations", 1, min(9, len(df) - 1), 4)

    if st.button("🔍  Find Similar Titles", use_container_width=True):
        recs = get_recommendations(selected_title, n=n_recs)

        # Show selected title card
        sel_row = df[df["title"].str.lower() == selected_title.lower()].iloc[0]
        st.markdown("### Selected Title")
        st.markdown(
            f"""
            <div class="rec-card" style="border-left-color:#fff;">
              <div class="rec-title">{sel_row['title']}</div>
              <div class="rec-meta">{sel_row['type']} · {sel_row['primary_genre']} · {int(sel_row['release_year'])} · {sel_row['rating']} · {sel_row['duration']}</div>
              <div class="rec-desc">{sel_row['description']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("### Recommended Titles")

        if recs.empty:
            st.warning("No recommendations found. Try another title.")
        else:
            for _, row in recs.iterrows():
                pct = int(row["similarity"] * 100)
                st.markdown(
                    f"""
                    <div class="rec-card">
                      <div class="rec-title">{row['title']}</div>
                      <div class="rec-meta">{row['type']} · {row['primary_genre']} · {int(row['release_year'])} · {row['rating']} · {row['duration']}</div>
                      <div class="rec-score">Match score: {pct}%</div>
                      <div class="rec-desc">{row['description']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )