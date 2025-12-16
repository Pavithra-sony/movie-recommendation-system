import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Placeholder poster ----
def fetch_poster_by_title(title):
    return "https://via.placeholder.com/300x450.png?text=Movie+Poster"

# ---- Load movies and compute similarity safely ----
def load_movies_and_similarity():
    try:
        movies_df = pd.read_csv('cleaned_movies.csv')
    except FileNotFoundError:
        st.error("cleaned_movies.csv not found! Upload it in the repo root.")
        return pd.DataFrame(), None

    # Ensure required columns exist
    for col in ['title', 'genre']:
        if col not in movies_df.columns:
            st.warning(f"Column '{col}' missing. Adding placeholder data.")
            movies_df[col] = "Unknown" if col == 'genre' else "No Title"

    # Compute TF-IDF matrix and similarity
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genre'].fillna(''))
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return movies_df, similarity_matrix

# ---- Recommendation function ----
def recommend(movie, movies_df, similarity_matrix):
    if similarity_matrix is None or movies_df.empty:
        return [], []

    if movie not in movies_df['title'].values:
        st.warning(f"Movie '{movie}' not found in dataset.")
        return [], []

    index = movies_df[movies_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])

    recommended_names = []
    recommended_posters = []

    for i in distances[1:6]:  # Top 5 recommendations
        movie_title = movies_df.iloc[i[0]]['title']
        recommended_names.append(movie_title)
        recommended_posters.append(fetch_poster_by_title(movie_title))

    return recommended_names, recommended_posters

# ---- Load data ----
movies, similarity = load_movies_and_similarity()

# ---- Streamlit UI ----
st.title("🎬 Movie Recommendation System")

if not movies.empty:
    selected_movie = st.selectbox("Select a movie", movies['title'].values)

    if st.button("Show Recommendation"):
        names, posters = recommend(selected_movie, movies, similarity)
        if names:
            cols = st.columns(5)
            for col, name, poster in zip(cols, names, posters):
                col.text(name)
                col.image(poster, use_container_width=True)
        else:
            st.info("No recommendations available.")
else:
    st.write("Movie data is not available. Check your CSV!")
