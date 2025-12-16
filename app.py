import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Safely fetch poster by movie title (placeholder)
def fetch_poster_by_title(title):
    return "https://via.placeholder.com/300x450.png?text=Movie+Poster"

# Load movie list and compute similarity dynamically
@st.cache_data
def load_movies_and_similarity():
    # Load CSV from repo
    movies_df = pd.read_csv('cleaned_movies.csv')  # must be in repo
    # Compute TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_df['genre'].fillna(''))  # adjust column name if needed
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return movies_df, similarity_matrix

# Recommend function
def recommend(movie, movies_df, similarity_matrix):
    index = movies_df[movies_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_title = movies_df.iloc[i[0]].title
        recommended_movie_names.append(movie_title)
        poster_url = fetch_poster_by_title(movie_title)
        recommended_movie_posters.append(poster_url)

    return recommended_movie_names, recommended_movie_posters

# Load data
movies, similarity = load_movies_and_similarity()

# Streamlit UI
st.header('🎬 Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie, movies, similarity)

    # Display in columns
    cols = st.columns(5)
    for col, name, poster in zip(cols, recommended_movie_names, recommended_movie_posters):
        col.text(name)
        col.image(poster, use_container_width=True)

