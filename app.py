import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load assets
cv = joblib.load('vectorizer.pkl')
movies_df = joblib.load('movies_df.pkl')

# Recompute similarity on-the-fly
count_matrix = cv.transform(movies_df['Combined_Features'])
cosine_sim = cosine_similarity(count_matrix)

# Streamlit UI
st.title("🎬 Movie Recommender")

def get_title_from_index(index):
    return movies_df.loc[index, 'names'].title()

def get_index_from_title(title):
    title = title.lower()
    try:
        return movies_df[movies_df['names'] == title].index[0]
    except IndexError:
        return None

movie_title = st.text_input("Enter a movie name")

if st.button("Recommend"):
    idx = get_index_from_title(movie_title)
    if idx is not None:
        similar = list(enumerate(cosine_sim[idx]))
        sorted_similar = sorted(similar, key=lambda x: x[1], reverse=True)[1:6]
        st.subheader("Top 5 similar movies:")
        for i in sorted_similar:
            st.write(get_title_from_index(i[0]))
    else:
        st.error("Movie not found.")