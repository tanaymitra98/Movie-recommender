import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# 🎯 Load model components
cv = joblib.load("vectorizer.pkl")
movies_df = joblib.load("movies_df.pkl")

# 🧠 Recompute cosine similarity from scratch
count_matrix = cv.transform(movies_df["Combined_Features"])
cosine_sim = cosine_similarity(count_matrix)

# 🔁 Helper functions
def get_title_from_index(index):
    return movies_df.loc[index, "names"].title()

def get_index_from_title(title):
    title = title.lower()
    try:
        return movies_df[movies_df["names"] == title].index[0]
    except IndexError:
        return None

# 🎬 Streamlit UI
st.set_page_config(page_title="Movie Recommender", layout="centered")
st.title("🎬 Movie Recommender System")

# Dropdown list of movie titles
movie_list = movies_df["names"].str.title().tolist()
movie_title = st.selectbox("Pick a movie:", sorted(movie_list))

# Recommend on button click
if st.button("Get Recommendations"):
    idx = get_index_from_title(movie_title)
    if idx is not None:
        similar_movies = list(enumerate(cosine_sim[idx]))
        sorted_similar = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

        st.subheader("Top 5 similar movies:")
        for i in sorted_similar:
            st.write(f"• {get_title_from_index(i[0])}")
    else:
        st.error("Movie not found.")
