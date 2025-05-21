import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from sklearn.preprocessing import RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Game Recommendation System",
    page_icon="🎮",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        color: #2E86C1;
        font-size: 3rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subtitle {
        color: #5499C7;
        text-align: center;
        margin-bottom: 2rem;
    }
    .game-card {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .game-image {
        width: 100%;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title">🎮 Game Recommendation System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Find your next favorite game based on your current favorites!</p>', unsafe_allow_html=True)

def get_game_image_url(game_title):
    """Get game image from Steam API"""
    try:
        # Search for the game
        search_url = f"https://store.steampowered.com/api/storesearch/?term={game_title}&l=english&cc=US"
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            if data.get('total') > 0:
                game = data['items'][0]
                return f"https://cdn.cloudflare.steamstatic.com/steam/apps/{game['id']}/header.jpg"
    except:
        pass
    # Return a default gaming image if no Steam image is found
    return "https://img.freepik.com/free-vector/video-game-controller-neon-sign-brick-wall-background_1262-11721.jpg"

@st.cache_data
def load_data():
    df = pd.read_csv('video_game_data(1).csv')
    return df

@st.cache_data
def preprocess_data(dataset):
    dataset['tags'] = dataset['tags'].apply(lambda x: " ".join(ast.literal_eval(x)))
    dataset['win'] = dataset['win'].map({True: 1, False: 0})
    dataset['mac'] = dataset['mac'].map({True: 1, False: 0})
    dataset['linux'] = dataset['linux'].map({True: 1, False: 0})
    dataset['rating'] = dataset['rating'].map({
        'Overwhelmingly Positive': 1,
        'Very Positive': 2,
        'Positive': 3,
        'Mostly Positive': 4,
        'Mixed': 5,
        'Mostly Negative': 6,
        'Negative': 7,
        'Very Negative': 8,
        'Overwhelmingly Negative': 9
    })
    dataset['user_reviews'] = np.log1p(dataset['user_reviews'])
    dataset['title'] = dataset['title'].str.lower()
    dataset = dataset[~dataset['title'].duplicated(keep=False)]
    return dataset

@st.cache_resource
def train_model(df):
    to_scale = ['positive_ratio', 'user_reviews']
    X = df[to_scale]
    rb = RobustScaler()
    X_scaled = rb.fit_transform(X)
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_tags = tfidf.fit_transform(df['tags'].fillna(''))
    
    combine = hstack([X_scaled, tfidf_tags])
    combine = combine.tocsr()
    
    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(combine)
    
    return combine, nn

def get_recommendations(combine, nn, df, game_title, top_games=5):
    try:
        game_index = df[df['title'] == game_title.lower()].index[0]
        dist, indices = nn.kneighbors(combine[game_index], n_neighbors=top_games + 1)
        searched = df[df['title'] == game_title][['title', 'positive_ratio', 'user_reviews', 'tags']]
        recommend = df.iloc[indices[0][1:]][['title', 'positive_ratio', 'user_reviews', 'tags']]
        result = pd.concat([searched, recommend])
        return result, True
    except:
        return None, False

def plot_similarity(combine, nn, df, game_title, top_games=10):
    try:
        game_index = df[df['title'] == game_title.lower()].index[0]
        dist, indices = nn.kneighbors(combine[game_index], n_neighbors=top_games + 1)
        score = 1 - dist[0][1:]
        titles = df.iloc[indices[0][1:]]['title']
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(titles[::-1], score[::-1])
        ax.set_xlabel('Similarity Score')
        ax.set_title('Similar Games')
        st.pyplot(fig)
        return True
    except:
        return False

# Load and preprocess data
with st.spinner('Loading game data...'):
    df = load_data()
    df = preprocess_data(df)
    combine, nn = train_model(df)

# Create sidebar
st.sidebar.header('Search Options')
search_type = st.sidebar.radio(
    "Search Type",
    ["Game Title", "Browse Popular Games"]
)

if search_type == "Game Title":
    # Create a search box with autocomplete
    game_titles = sorted(df['title'].unique())
    search_query = st.text_input("Enter a game title:", "")
    
    # Filter suggestions based on user input
    suggestions = [title for title in game_titles if search_query.lower() in title.lower()]
    
    if suggestions and search_query:
        selected_game = st.selectbox("Did you mean:", suggestions)
        
        if st.button("Get Recommendations"):
            recommendations, success = get_recommendations(combine, nn, df, selected_game)
            
            if success:
                st.subheader("Recommended Games")
                
                # Create three columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display recommendations in a nice format
                    for idx, row in recommendations.iterrows():
                        with st.container():
                            st.markdown(f"### {row['title'].title()}")
                            col_img, col_info = st.columns([1, 2])
                            with col_img:
                                st.image(get_game_image_url(row['title']), use_column_width=True)
                            with col_info:
                                st.markdown(f"**Positive Ratio:** {row['positive_ratio']:.2f}")
                                st.markdown(f"**User Reviews:** {int(np.exp(row['user_reviews']) - 1)}")
                                st.markdown(f"**Tags:** {row['tags']}")
                            st.markdown("---")
                
                with col2:
                    st.subheader("Similarity Graph")
                    plot_similarity(combine, nn, df, selected_game)
            else:
                st.error("Sorry, we couldn't find that game in our database.")

else:  # Browse Popular Games
    st.subheader("Popular Games")
    # Sort by number of user reviews and positive ratio
    popular_games = df.sort_values(by=['user_reviews', 'positive_ratio'], ascending=[False, False]).head(10)
    
    for idx, row in popular_games.iterrows():
        with st.container():
            st.markdown(f"### {row['title'].title()}")
            col_img, col_info = st.columns([1, 2])
            with col_img:
                st.image(get_game_image_url(row['title']), use_column_width=True)
            with col_info:
                st.markdown(f"**Positive Ratio:** {row['positive_ratio']:.2f}")
                st.markdown(f"**User Reviews:** {int(np.exp(row['user_reviews']) - 1)}")
                st.markdown(f"**Tags:** {row['tags']}")
            if st.button(f"Get Similar Games to {row['title'].title()}", key=f"popular_{idx}"):
                recommendations, success = get_recommendations(combine, nn, df, row['title'])
                if success:
                    st.markdown("#### Similar Games:")
                    plot_similarity(combine, nn, df, row['title'])
                    for rec_idx, rec_row in recommendations.iloc[1:].iterrows():
                        col_rec_img, col_rec_info = st.columns([1, 2])
                        with col_rec_img:
                            st.image(get_game_image_url(rec_row['title']), use_column_width=True)
                        with col_rec_info:
                            st.markdown(f"- {rec_row['title'].title()}")
                            st.markdown(f"**Tags:** {rec_row['tags']}")
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and Machine Learning")
