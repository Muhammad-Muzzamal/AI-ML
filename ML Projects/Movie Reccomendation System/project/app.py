import streamlit as st
import pickle

movies_list = pickle.load(open("movies.pkl", "rb"))
movies_list = movies_list["title"].values



st.title("Movie Recommender System")



selected_movies_name = st.selectbox(
    "How would you like to be contacted?",
    movies_list
)


if st.button("Recommend"):
    recommend(selected_movies_name)
    st.write(selected_movies_name)