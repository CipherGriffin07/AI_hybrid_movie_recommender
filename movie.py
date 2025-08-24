# streamlit_ai_recommender_v2.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------
# Step 1: Movie Dataset
# -----------------------------
data = {
    'MovieID': [1,2,3,4,5,6,7,8],
    'Title': ['The Matrix','Titanic','Inception','Avatar','The Godfather','Interstellar','Gladiator','Joker'],
    'Genre': ['Action','Romance','Action','Action','Crime','Sci-Fi','Action','Crime'],
    'Director': ['Wachowski','Cameron','Nolan','Cameron','Coppola','Nolan','Scott','Todd Phillips'],
    'Cast': ['Keanu Reeves','Leonardo DiCaprio','Leonardo DiCaprio','Sam Worthington','Marlon Brando','Matthew McConaughey','Russell Crowe','Joaquin Phoenix'],
    'Year': [1999,1997,2010,2009,1972,2014,2000,2019],
    'IMDB_Rating': [8.7,7.8,8.8,7.8,9.2,8.6,8.5,8.4],
    'User_Rating': [9,8,9,8,10,9,8,9],
    'Description': [
        "A hacker discovers reality is a simulation and joins a rebellion.",
        "A tragic love story on the ill-fated Titanic ship.",
        "A thief who enters dreams to steal secrets is given a complex mission.",
        "Humans colonize Pandora and face conflicts with the native species.",
        "The rise and fall of a mafia family in America.",
        "A team travels through a wormhole to find a new habitable planet.",
        "A Roman general seeks revenge against a corrupt emperor.",
        "A mentally troubled man descends into chaos in Gotham City."
    ]
}

movies = pd.DataFrame(data)

# -----------------------------
# Step 2: AI Embeddings
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()
movies['Embedding'] = movies['Description'].apply(lambda x: model.encode(x))
content_matrix = np.vstack(movies['Embedding'].values)

# Collaborative numeric features
scaler = StandardScaler()
numeric_features = scaler.fit_transform(movies[['IMDB_Rating','User_Rating','Year']])
collab_similarity = cosine_similarity(numeric_features)

# -----------------------------
# Step 3: Hybrid Recommendation Function
# -----------------------------
def hybrid_recommend(user_input, top_n=3, alpha=0.7):
    # Content similarity
    if user_input in movies['Title'].values:
        idx = movies[movies['Title'] == user_input].index[0]
        content_sim = cosine_similarity([content_matrix[idx]], content_matrix)[0]
    else:
        user_emb = model.encode(user_input)
        content_sim = cosine_similarity([user_emb], content_matrix)[0]

    hybrid_sim = alpha * content_sim + (1-alpha) * collab_similarity.mean(axis=0)

    sim_scores = list(enumerate(hybrid_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [i for i in sim_scores if movies.iloc[i[0]]['Title'] != user_input][:top_n]

    movie_indices = [i[0] for i in sim_scores]
    recommended = movies.iloc[movie_indices]

    # Visualization
    scores = [i[1] for i in sim_scores]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(recommended['Title'], scores, color='lightcoral')
    ax.set_ylabel("Hybrid Similarity Score")
    ax.set_title(f"Top {top_n} Recommendations for '{user_input}'")
    st.pyplot(fig)

    return recommended[['Title','Genre','Director','IMDB_Rating','User_Rating','Description']]

# -----------------------------
# Step 4: Streamlit UI
# -----------------------------
st.title("🎬 AI-Powered Hybrid Movie Recommender")
st.write("Enter a movie title or describe your movie preference in a sentence.")

# Sidebar: Trending / Top Picks
st.sidebar.header("🔥 Top Picks / Trending Movies")
top_picks = movies.sort_values(by=['IMDB_Rating','User_Rating'], ascending=False).head(3)
st.sidebar.table(top_picks[['Title','Genre','IMDB_Rating','User_Rating']])

user_input = st.text_input("Movie name or description", "")
top_n = st.slider("Number of Recommendations", min_value=1, max_value=5, value=3)
alpha = st.slider("Weight for content-based similarity (0-1)", min_value=0.0, max_value=1.0, value=0.7)

if st.button("Get Recommendations") and user_input.strip() != "":
    recommendations = hybrid_recommend(user_input, top_n, alpha)
    st.write("### Recommended Movies:")
    st.dataframe(recommendations.reset_index(drop=True))
