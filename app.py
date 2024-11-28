import streamlit as st
import pandas as pd
from MusicRecommender import MusicRecommender  # Make sure this matches your class file name

# Set page configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state for selected songs
if 'selected_songs' not in st.session_state:
    st.session_state.selected_songs = []

# Load data and initialize recommender
@st.cache_data
def load_data():
    return pd.read_csv('song_dataset.csv')

@st.cache_resource
def init_recommender(df):
    recommender = MusicRecommender(df)
    recommender.fit()
    return recommender

# Main app
def main():
    st.title("🎵 Music Recommendation System")
    st.write("Select songs you like and get personalized recommendations!")

    # Load data and initialize recommender
    df = load_data()
    recommender = init_recommender(df)

    # Create two columns for layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Select Your Songs")
        # Create song selection dropdown
        unique_songs = df[['title', 'artist_name']].drop_duplicates()
        song_options = [f"{row['title']} by {row['artist_name']}" 
                       for _, row in unique_songs.iterrows()]
        
        selected_song = st.selectbox(
            "Choose a song:",
            [""] + song_options,
            key='song_select'
        )

        if selected_song and selected_song not in st.session_state.selected_songs:
            if st.button("Add Song"):
                st.session_state.selected_songs.append(selected_song)

        # Display selected songs
        st.write("### Your Selected Songs:")
        for i, song in enumerate(st.session_state.selected_songs):
            col1.write(f"{i+1}. {song}")
            if st.button(f"Remove {i}", key=f"remove_{i}"):
                st.session_state.selected_songs.pop(i)
                st.experimental_rerun()

    with col2:
        st.subheader("Recommendations")
        
        diversity_weight = st.slider(
            "Diversity Level",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.1
        )

        if st.button("Get Recommendations") and st.session_state.selected_songs:
            # Create a temporary user profile from selected songs
            temp_user_songs = []
            for song_str in st.session_state.selected_songs:
                title = song_str.split(" by ")[0]
                artist = song_str.split(" by ")[1]
                matching_songs = df[
                    (df['title'] == title) & 
                    (df['artist_name'] == artist)
                ]['song'].values
                if len(matching_songs) > 0:
                    temp_user_songs.append(matching_songs[0])

            # Get recommendations
            recommendations = recommender.get_recommendations(
                df['user'].iloc[0],  # Use first user as template
                n_recommendations=5,
                diversity_weight=diversity_weight
            )

            # Display recommendations
            st.write("### Recommended Songs:")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec['title']} by {rec['artist']} (Score: {rec['score']:.2f})")
                st.write(f"   Genre: {rec['genre']}")

if __name__ == "__main__":
    main()