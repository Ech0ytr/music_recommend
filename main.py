from flask import Flask, request, jsonify, render_template, session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)
app.secret_key = 'secret-key'  

def train_model():
    df = pd.read_csv('attached_assets/songs_normalize.csv')

    # Handle potential NaN values
    df = df.fillna(0)

    # Create label encoders for categorical features
    le_genre = LabelEncoder()
    df['genre_encoded'] = le_genre.fit_transform(df['genre'])

    # Feature Engineering
    features = ['year', 'duration_ms', 'popularity', 'danceability', 'energy', 
                'tempo', 'genre_encoded']
    X = df[features]
    y = df['song']  

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    return model, le_genre, df

model, le_genre, songs_df = train_model()

def get_unique_genres():
    return sorted(list(set([genre.strip() for genres in songs_df['genre'].unique() 
                          for genre in str(genres).split(',')])))

@app.route('/')
def home():
    genres = get_unique_genres()
    years = sorted(songs_df['year'].unique())
    return render_template('index.html', genres=genres, years=years)

@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if data['username'] and data['password']:
        session['username'] = data['username']
        return jsonify({'message': 'Registration successful'})
    return jsonify({'error': 'Invalid credentials'}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    if data['username'] and data['password']:
        session['username'] = data['username']
        return jsonify({'message': 'Login successful'})
    return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'username' not in session:
        return jsonify({'error': 'Please login first'}), 401

    data = request.get_json()
    song = data.get('song', '')
    genre = data.get('genre', '')
    year = data.get('year', 0)
    duration = data.get('duration', 0)
    tempo = data.get('tempo', 0)


    # Find similar songs using model
    input_song = songs_df[songs_df['song'].str.lower() == song.lower()]
    if len(input_song) > 0:
        features = ['year', 'duration_ms', 'popularity', 'danceability', 
                   'energy', 'tempo', 'genre_encoded']
        input_features = input_song[features].iloc[0]

        # Get similar songs using the model
        similarities = model.predict_proba([input_features])[0]
        similar_indices = np.argsort(similarities)[-5:][::-1]
        recommendations = songs_df.iloc[similar_indices]['song'].tolist()
    else:
        # Fallback to genre and year based filtering
        filtered_songs = songs_df[
            (songs_df['genre'].str.contains(genre, case=False, na=False)) &
            (songs_df['year'] >= int(year)-5) & 
            (songs_df['year'] <= int(year)+5)
        ]
        recommendations = filtered_songs.sample(n=min(5, len(filtered_songs)))['song'].tolist()

    return jsonify({
        'recommendations': recommendations,
        'song': song
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)