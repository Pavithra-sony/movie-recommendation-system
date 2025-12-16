import pickle
import streamlit as st


# Safely fetch poster by movie title from TMDb
def fetch_poster_by_title(title):
    return "https://via.placeholder.com/300x450.png?text=Movie+Poster"

# Load movie list
@st.cache_resource
def load_movies():
    return pickle.load(open('movieslist.pkl', 'rb'))  # Change filename if needed

# Load similarity matrix
@st.cache_resource
def load_similarity():
    return pickle.load(open('similarity.pkl', 'rb'))  # Change filename if needed

# Recommend function
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1:6]:
        movie_title = movies.iloc[i[0]].title
        recommended_movie_names.append(movie_title)
        poster_url = fetch_poster_by_title(movie_title)
        recommended_movie_posters.append(poster_url)

    return recommended_movie_names, recommended_movie_posters

# Load data
movies = load_movies()
similarity = load_similarity()

# Streamlit UI
st.header('🎬 Movie Recommender System')

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie", movie_list)

# Show recommendations when the button is clicked
if st.button('Show Recommendation'):
    recommended_movie_names, recommended_movie_posters = recommend(selected_movie)

    # Display recommendations in columns (as you requested)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.text(recommended_movie_names[0])
        st.image(recommended_movie_posters[0], use_container_width=True)

    with col2:
        st.text(recommended_movie_names[1])
        st.image(recommended_movie_posters[1], use_container_width=True)

    with col3:
        st.text(recommended_movie_names[2])
        st.image(recommended_movie_posters[2], use_container_width=True)

    with col4:
        st.text(recommended_movie_names[3])
        st.image(recommended_movie_posters[3], use_container_width=True)

    with col5:
        st.text(recommended_movie_names[4])
        st.image(recommended_movie_posters[4], use_container_width=True)
