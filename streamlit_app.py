
# ================================================================
# STREAMLIT APP - CRS DEMO INTERFACE
# Save this as: streamlit_app.py
# Run with: streamlit run streamlit_app.py
# ================================================================
# Before running, make sure these files exist in the same folder:
#   - rmp_final_with_features.csv
#   - best_model_lr.joblib
#   - tfidf_vectorizer.joblib
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# ── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Feedback CRS",
    page_icon="🎓",
    layout="wide"
)

# ── Load data and models ─────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv('data/rmp_final_with_features.csv')

@st.cache_resource
def load_models():
    model  = joblib.load('models/best_model_lr.joblib')
    tfidf  = joblib.load('models/tfidf_vectorizer.joblib')
    return model, tfidf

@st.cache_resource
def build_rag_index(df):
    tfidf_rag = TfidfVectorizer(stop_words='english', max_features=8000, ngram_range=(1,2))
    matrix    = tfidf_rag.fit_transform(df['clean_comments'].fillna(''))
    return tfidf_rag, matrix

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
    
def summarize_reviews(reviews_df):
    total      = len(reviews_df)
    pos_pct    = (reviews_df['sentiment'] == 'positive').mean() * 100
    neg_pct    = (reviews_df['sentiment'] == 'negative').mean() * 100
    neu_pct    = (reviews_df['sentiment'] == 'neutral').mean() * 100
    top_aspect = reviews_df['primary_aspect'].value_counts().index[0] if 'primary_aspect' in reviews_df.columns else 'general'
    avg_rating = reviews_df['student_star'].mean() if 'student_star' in reviews_df.columns else None

    summary = f"📝 **Auto Summary:** Based on {total} retrieved reviews, "
    summary += f"**{pos_pct:.0f}%** were positive, "
    summary += f"**{neg_pct:.0f}%** negative, and "
    summary += f"**{neu_pct:.0f}%** neutral. "
    summary += f"The most discussed topic was **{top_aspect.upper()}**. "
    if avg_rating and not np.isnan(avg_rating):
        summary += f"Average rating: **{avg_rating:.1f}/5**."
    return summary
    
df    = load_data()
model, tfidf = load_models()
tfidf_rag, comment_matrix = build_rag_index(df)

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/graduation-cap.png", width=80)
st.sidebar.title("🎓 Student Feedback CRS")
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "💬 Chatbot",
    "🔮 Sentiment Predictor",
    "📊 Analytics",
    "🏆 Professor Recommender"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df):,} reviews")
st.sidebar.markdown(f"**Departments:** {df['dept_category'].nunique()}")
st.sidebar.markdown(f"**Professors:** {df['professor_name'].nunique():,}")

# ================================================================
# PAGE 1: OVERVIEW
# ================================================================
if page == "🏠 Overview":
    st.title("🎓 Intelligent Student Feedback Analysis System")
    st.markdown("### Conversational Recommender System (CRS)")
    st.markdown("""
    This system analyzes student reviews from **RateMyProfessor** using:
    - 🤖 **Sentiment Classification** (Logistic Regression, SVM, Naive Bayes, Random Forest, Gradient Boosting)
    - 🔍 **Aspect Extraction** using SpaCy NLP
    - 📦 **Clustering** to discover feedback themes (KMeans, DBSCAN, Agglomerative)
    - 💬 **RAG Chatbot** for natural language Q&A over student reviews
    - 🏆 **Professor Recommender** using collaborative filtering (SVD, KNN, NMF)
    """)

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews",    f"{len(df):,}")
    col2.metric("Departments",      df['dept_category'].nunique())
    col3.metric("Unique Professors",df['professor_name'].nunique())
    col4.metric("Avg Rating",       f"{df['student_star'].mean():.2f}/5")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        sent_counts = df['sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#2ecc71', '#e74c3c', '#f39c12']
        ax.pie(sent_counts.values, labels=sent_counts.index,
               colors=colors, autopct='%1.1f%%', startangle=140)
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

    with col2:
        st.subheader("Reviews by Department")
        dept_counts = df['dept_category'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        dept_counts.plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
        ax.set_xlabel("Count")
        st.pyplot(fig)

    st.subheader("Sample Reviews")
    st.dataframe(
        df[['professor_name', 'dept_category', 'sentiment', 'comments', 'primary_aspect']]
        .sample(10, random_state=42)
        .reset_index(drop=True),
        use_container_width=True
    )

# ================================================================
# PAGE 2: CHATBOT
# ================================================================
elif page == "💬 Chatbot":
    st.title("💬 RAG-Powered Chatbot")
    st.markdown("Ask anything about student feedback. The chatbot retrieves real reviews to answer your question.")

    # Chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Hi! I'm your Student Feedback Assistant. Ask me anything about student reviews!\n\nTry:\n- *What do students complain about exams?*\n- *Recommend a good mathematics professor*\n- *What do students say about grading in English?*"
        })

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Filters
    with st.expander("🔧 Filter Options"):
        col1, col2, col3 = st.columns(3)
        dept_filter = col1.selectbox("Department", ["All"] + sorted(df['dept_category'].unique().tolist()))
        sent_filter = col2.selectbox("Sentiment", ["All", "positive", "neutral", "negative"])
        top_k       = col3.slider("Number of reviews to retrieve", 3, 10, 5)

        dept_f = None if dept_filter == "All" else dept_filter
        sent_f = None if sent_filter == "All" else sent_filter

    # Chat input
    if user_input := st.chat_input("Ask about student feedback..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # ── FALLBACK DETECTION ────────────────────────────────────────
        def is_off_topic(query):
            q = query.lower().strip()
            
            off_topic_patterns = [
                # greetings
                r'\b(hi|hello|hey|howdy|sup|whats up|what\'s up)\b',
                # weather
                r'\b(weather|sunny|raining|temperature|forecast|cold|hot|humid)\b',
                # time/date
                r'\b(time|date|today|tomorrow|yesterday|clock|what day)\b',
                # food
                r'\b(food|hungry|eat|lunch|dinner|breakfast|pizza|burger)\b',
                # jokes
                r'\b(joke|funny|laugh|humor|meme)\b',
                # sports
                r'\b(football|basketball|soccer|cricket|sports|game|match|score)\b',
                # politics
                r'\b(politics|president|election|government|minister|party)\b',
                # tech/general
                r'\b(iphone|android|computer|internet|google|facebook|instagram)\b',
                # math/calculations
                r'\b(calculate|what is \d|solve|equation|\d+\s*[\+\-\*\/]\s*\d+)\b',
                # personal
                r'\b(your name|who are you|how old|are you human|are you ai|are you a bot)\b',
            ]
            
            # Check if query has ANY education-related words
            education_keywords = [
                'professor', 'prof', 'teacher', 'class', 'course', 'lecture',
                'exam', 'test', 'grade', 'homework', 'assignment', 'student',
                'department', 'math', 'english', 'history', 'biology', 'psychology',
                'syllabus', 'attendance', 'workload', 'recommend', 'review', 'rating'
            ]
            
            has_education = any(kw in q for kw in education_keywords)
            if has_education:
                return False  # relevant, don't flag as off-topic
            
            for pattern in off_topic_patterns:
                if re.search(pattern, q):
                    return True
            
            # If query is very short and has no education context
            if len(q.split()) <= 3 and not has_education:
                return True
                
            return False

        FALLBACK_RESPONSES = {
            'greeting':   "👋 Hi there! I'm your **Student Feedback Assistant**. I can help you with:\n- 🔍 Finding student reviews\n- 🏆 Recommending professors\n- 📊 Analyzing feedback by department\n\nTry asking: *What do students say about math professors?*",
            'weather':    "⛅ I don't have weather data, but I can tell you how students *feel* about their professors! Try asking: *What are the best biology professors?*",
            'time':       "🕐 I don't track time, but I can help you find the right professor before registration! Try: *Recommend a good history professor.*",
            'off_topic':  "🤖 I'm specialized in **student feedback and professor reviews**. I can't help with that, but I can:\n- Find reviews for a specific department\n- Recommend top-rated professors\n- Summarize student complaints or praise\n\nWhat would you like to know about professors or courses?",
        }

        def get_fallback(query):
            q = query.lower()
            if re.search(r'\b(hi|hello|hey|howdy)\b', q):
                return FALLBACK_RESPONSES['greeting']
            if re.search(r'\b(weather|sunny|raining|temperature)\b', q):
                return FALLBACK_RESPONSES['weather']
            if re.search(r'\b(time|date|today|tomorrow|clock)\b', q):
                return FALLBACK_RESPONSES['time']
            return FALLBACK_RESPONSES['off_topic']

        # ── MAIN LOGIC ────────────────────────────────────────────────
        if is_off_topic(user_input):
            response = get_fallback(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

        else:
            # Original retrieval logic stays exactly the same
            query_clean = clean_text(user_input)
            query_vec   = tfidf_rag.transform([query_clean])
            sims        = cosine_similarity(query_vec, comment_matrix).flatten()

            result_df = df.copy()
            result_df['similarity'] = sims
            if sent_f:
                result_df = result_df[result_df['sentiment'] == sent_f]
            if dept_f:
                result_df = result_df[result_df['dept_category'] == dept_f]

            top_reviews = result_df.nlargest(top_k, 'similarity')

            if top_reviews.empty:
                response = "❌ No relevant reviews found. Try adjusting your filters or rephrasing."
            else:
                sent_dist = top_reviews['sentiment'].value_counts()
                response_parts = [f"📚 Found **{len(top_reviews)}** relevant reviews.\n"]
                response_parts.append("**📊 Sentiment breakdown:**")
                for s, c in sent_dist.items():
                    emoji = '✅' if s == 'positive' else ('⚠️' if s == 'neutral' else '❌')
                    response_parts.append(f"- {emoji} {s.capitalize()}: {c}")
                response_parts.append("\n**💬 Top matching reviews:**")
                for i, (_, row) in enumerate(top_reviews.head(3).iterrows(), 1):
                    emoji = '✅' if row['sentiment'] == 'positive' else ('⚠️' if row['sentiment'] == 'neutral' else '❌')
                    response_parts.append(
                        f"\n**{i}.** [{emoji} {row['sentiment'].upper()}] *{row['dept_category']}*\n"
                        f"> {row['comments'][:200]}{'...' if len(str(row['comments'])) > 200 else ''}"
                    )
                response_parts.append("\n**💬 Top matching reviews:**")
                response_parts.append("\n" + summarize_reviews(top_reviews))    
                response = '\n'.join(response_parts)

            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

            with st.expander("📋 View all retrieved reviews"):
                st.dataframe(
                    top_reviews[['professor_name', 'dept_category', 'sentiment',
                                  'primary_aspect', 'similarity', 'comments']]
                    .reset_index(drop=True),
                    use_container_width=True
                )

# ================================================================
# PAGE 3: SENTIMENT PREDICTOR
# ================================================================
elif page == "🔮 Sentiment Predictor":
    st.title("🔮 Sentiment Predictor")
    st.markdown("Enter a student review and the model will predict its sentiment.")

    user_text = st.text_area(
        "Enter a student review:",
        placeholder="e.g., The professor explains everything clearly but the exams are really hard...",
        height=150
    )

    if st.button("🔍 Predict Sentiment", type="primary"):
        if user_text.strip():
            cleaned = clean_text(user_text)
            vec_tfidf = tfidf.transform([cleaned])
        
            from scipy.sparse import hstack, csr_matrix
        
            word_count      = len(cleaned.split())
            avg_word_len    = np.mean([len(w) for w in cleaned.split()]) if cleaned.split() else 0
            exclamations    = user_text.count('!')
            questions       = user_text.count('?')
            upper_ratio     = sum(1 for c in user_text if c.isupper()) / max(len(user_text), 1)
            
            POSITIVE_WORDS = {'great','excellent','amazing','wonderful','best','good',
                              'helpful','clear','fantastic','love','awesome','brilliant'}
            NEGATIVE_WORDS = {'bad','terrible','awful','worst','boring','hard','difficult',
                              'confusing','unfair','horrible','waste','useless','poor'}
            words     = set(cleaned.lower().split())
            pos_count = len(words & POSITIVE_WORDS)
            neg_count = len(words & NEGATIVE_WORDS)
            sent_ratio = (pos_count - neg_count) / (word_count + 1)
            numeric_feats = np.array([[
                word_count, avg_word_len, exclamations, questions, upper_ratio,
                pos_count, neg_count, sent_ratio,
                0,  # has_person_mention
                1,  # num_aspects (default 1)
                0,  # diff_index
                0,  # student_difficult
                0   # would_take_agains
            ]])
            vec = hstack([vec_tfidf, csr_matrix(numeric_feats)])
            pred    = model.predict(vec)[0]

            # Try to get probabilities
            try:
                proba = model.predict_proba(vec)[0]
                classes = model.classes_
            except:
                proba   = None
                classes = None

            # Display result
            color   = {'positive': '🟢', 'neutral': '🟡', 'negative': '🔴'}
            st.markdown(f"### Predicted Sentiment: {color.get(pred, '⚪')} **{pred.upper()}**")

            if proba is not None:
                st.markdown("#### Confidence Scores:")
                for cls, prob in zip(classes, proba):
                    st.progress(float(prob), text=f"{cls.capitalize()}: {prob*100:.1f}%")

            # Also run aspect extraction
            ASPECT_KEYWORDS = {
                'exams':     ['exam', 'test', 'quiz', 'midterm', 'final'],
                'grading':   ['grade', 'grading', 'rubric', 'score', 'mark'],
                'workload':  ['homework', 'assignment', 'work', 'reading', 'load'],
                'lectures':  ['lecture', 'class', 'explain', 'teaching', 'slide'],
                'professor': ['professor', 'teacher', 'prof', 'instructor'],
            }
            doc    = nlp(cleaned[:500])
            found  = []
            for token in doc:
                for aspect, kws in ASPECT_KEYWORDS.items():
                    if token.lemma_ in kws and aspect not in found:
                        found.append(aspect)

            if found:
                st.markdown(f"**🎯 Detected Aspects:** {', '.join(found)}")

            # Show similar reviews
            st.markdown("---")
            st.markdown("**📋 Similar reviews in dataset:**")
            q_vec = tfidf_rag.transform([cleaned])
            sims  = cosine_similarity(q_vec, comment_matrix).flatten()
            top   = df.copy()
            top['sim'] = sims
            top = top[top['sentiment'] == pred].nlargest(3, 'sim')
            for _, row in top.iterrows():
                st.markdown(f"> *{row['comments'][:200]}*")
                st.caption(f"Dept: {row['dept_category']} | Sentiment: {row['sentiment']}")
        else:
            st.warning("Please enter a review first.")

# ================================================================
# PAGE 4: ANALYTICS
# ================================================================
elif page == "📊 Analytics":
    st.title("📊 Analytics Dashboard")

    tab1, tab2, tab3 = st.tabs(["Classification Results", "Clustering", "Aspect Analysis"])

    with tab1:
        st.subheader("Model Performance Comparison")

        # Load saved summary if available
        try:
            with open('outputs/project_summary.json') as f:
                summary = json.load(f)
            clf = summary['classification']
            perf_df = pd.DataFrame(clf).T
            perf_df.index.name = 'Model'
            st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            perf_df[['accuracy', 'f1_macro', 'f1_weighted']].plot(
                kind='bar', ax=ax, edgecolor='black'
            )
            ax.set_title('Model Comparison')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            ax.legend(['Accuracy', 'F1 Macro', 'F1 Weighted'])
            plt.xticks(rotation=20, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        except:
            st.info("Run the main notebook first to generate project_summary.json")

    with tab2:
        st.subheader("Cluster Analysis")

        if 'kmeans_cluster' in df.columns and 'cluster_name' in df.columns:
            cluster_name_map = df.groupby('kmeans_cluster')['cluster_name'].first().to_dict()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Reviews per Cluster**")
                cluster_counts = df['kmeans_cluster'].value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(6, 4))
                cluster_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_xticklabels(
                    [f"C{i}\n{cluster_name_map.get(i,'')[:12]}" for i in cluster_counts.index],
                    rotation=30, ha='right', fontsize=8
                )
                st.pyplot(fig)

            with col2:
                st.markdown("**Sentiment per Cluster**")
                ct = pd.crosstab(df['kmeans_cluster'], df['sentiment'], normalize='index') * 100
                fig, ax = plt.subplots(figsize=(6, 4))
                ct.plot(kind='bar', stacked=True, ax=ax,
                        color=['#e74c3c','#f39c12','#2ecc71'])
                ax.set_ylabel("Percentage")
                ax.set_xlabel("Cluster")
                plt.xticks(rotation=30)
                st.pyplot(fig)

            st.markdown("**Cluster Descriptions**")
            for cluster_id, name in cluster_name_map.items():
                with st.expander(f"Cluster {cluster_id}: {name}"):
                    cluster_df = df[df['kmeans_cluster'] == cluster_id]
                    st.write(f"Reviews: {len(cluster_df)}")
                    st.write(f"Sentiment: {cluster_df['sentiment'].value_counts().to_dict()}")
                    st.write("Sample review:")
                    st.info(cluster_df['comments'].iloc[0][:300])
        else:
            st.info("Run the main notebook first to generate cluster labels.")

    with tab3:
        st.subheader("Aspect Analysis")

        if 'primary_aspect' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Aspect Distribution**")
                aspect_counts = pd.Series(
                    [a for aspects in df['aspects'].apply(
                        lambda x: eval(x) if isinstance(x, str) else x
                    ) for a in aspects]
                ).value_counts()
                fig, ax = plt.subplots(figsize=(6, 4))
                aspect_counts.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='black')
                ax.set_xlabel("Aspect")
                ax.set_ylabel("Count")
                plt.xticks(rotation=30, ha='right')
                st.pyplot(fig)

            with col2:
                st.markdown("**Aspect vs Sentiment Heatmap**")
                asp_sent = pd.crosstab(df['primary_aspect'], df['sentiment'])
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.heatmap(asp_sent, annot=True, fmt='d', cmap='YlOrRd', ax=ax)
                ax.set_title("Aspect × Sentiment")
                st.pyplot(fig)

# ================================================================
# PAGE 5: PROFESSOR RECOMMENDER
# ================================================================
elif page == "🏆 Professor Recommender":
    st.title("🏆 Professor Recommender")
    st.markdown("Find the best professors based on student ratings and sentiment.")

    col1, col2 = st.columns([1, 2])

    with col1:
        dept_choice  = st.selectbox("Filter by Department", ["All"] + sorted(df['dept_category'].unique().tolist()))
        top_n        = st.slider("Number of recommendations", 3, 10, 5)
        min_reviews  = st.slider("Minimum reviews required", 2, 10, 3)
        sort_by      = st.selectbox("Sort by", ["Average Rating", "% Positive Reviews", "Number of Reviews"])

    with col2:
        dept_f = None if dept_choice == "All" else dept_choice
        prof_summary = df.copy()

        if dept_f:
            prof_summary = prof_summary[prof_summary['dept_category'] == dept_f]

        prof_summary = prof_summary.groupby('professor_name').agg(
            avg_rating    = ('student_star', 'mean'),
            num_reviews   = ('student_star', 'count'),
            dept          = ('dept_category', 'first'),
            school        = ('school_name', 'first'),
            pct_positive  = ('sentiment', lambda x: (x == 'positive').mean() * 100),
            pct_negative  = ('sentiment', lambda x: (x == 'negative').mean() * 100),
        ).reset_index()

        prof_summary = prof_summary[prof_summary['num_reviews'] >= min_reviews]

        sort_col_map = {
            "Average Rating":       "avg_rating",
            "% Positive Reviews":   "pct_positive",
            "Number of Reviews":    "num_reviews"
        }
        prof_summary = prof_summary.sort_values(sort_col_map[sort_by], ascending=False).head(top_n)

        prof_summary['avg_rating']   = prof_summary['avg_rating'].round(2)
        prof_summary['pct_positive'] = prof_summary['pct_positive'].round(1)
        prof_summary['pct_negative'] = prof_summary['pct_negative'].round(1)

        st.dataframe(
            prof_summary[['professor_name', 'dept', 'school',
                           'avg_rating', 'pct_positive', 'pct_negative', 'num_reviews']]
            .rename(columns={
                'professor_name': 'Professor',
                'dept':           'Department',
                'school':         'School',
                'avg_rating':     'Avg Rating',
                'pct_positive':   '% Positive',
                'pct_negative':   '% Negative',
                'num_reviews':    '# Reviews'
            })
            .reset_index(drop=True)
            .style.background_gradient(subset=['Avg Rating'], cmap='Greens')
                  .background_gradient(subset=['% Positive'], cmap='Greens')
                  .background_gradient(subset=['% Negative'], cmap='Reds_r'),
            use_container_width=True
        )

    # Click to see professor reviews
    st.markdown("---")
    st.subheader("📋 View Professor Reviews")
    selected_prof = st.selectbox("Select a professor", sorted(df['professor_name'].unique()))

    if selected_prof:
        prof_reviews = df[df['professor_name'] == selected_prof][
            ['comments', 'sentiment', 'dept_category', 'primary_aspect', 'student_star']
        ].reset_index(drop=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Reviews", len(prof_reviews))
        col2.metric("Avg Rating", f"{prof_reviews['student_star'].mean():.2f}/5")
        col3.metric("% Positive", f"{(prof_reviews['sentiment']=='positive').mean()*100:.0f}%")

        st.dataframe(prof_reviews, use_container_width=True)
