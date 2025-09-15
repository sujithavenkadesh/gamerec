# Game Recommendation System

A machine learning-based game recommendation system that suggests similar games based on user preferences and game characteristics. The system uses collaborative filtering and content-based filtering to provide personalized game recommendations.

## Features

- Search for games by title
- Browse popular games
- Get personalized game recommendations
- View game similarity scores
- See game details including ratings and user reviews
- Visual representation of game similarities

## Installation & Setup

1. Clone the repository:
```bash
git clone https://github.com/vsiva763-git/gamerec.git
cd gamerec
```

2. Create a virtual environment and activate it:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
pip install streamlit
```

4. Run the application:
```bash
streamlit run app.py
```

5. Open your web browser and go to:
```
http://localhost:8501
```

## Usage

1. **Search by Game Title:**
   - Select "Game Title" from the sidebar
   - Type a game name in the search box
   - Choose from the suggestions
   - Click "Get Recommendations"

2. **Browse Popular Games:**
   - Select "Browse Popular Games" from the sidebar
   - Browse through the list of popular games
   - Click "Get Similar Games" for any game you're interested in

## Required Data Files

Make sure you have the following files in your project directory:
- `app.py`: The main application file
- `video_game_data(1).csv`: The dataset file containing game information

## Dependencies

The main dependencies are:
- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- requests

For the complete list of dependencies, see `requirements.txt`

## Screenshots

![Game Search Interface](game_recommendation_20250521_090405.png)

## Contributing

Feel free to open issues and pull requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details.
