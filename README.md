# Music Recommendation System
A web-based music recommendation system built with Flask that allows users to register, log in, and receive song recommendations based on their input. The app uses a preprocessed music dataset and simple filtering techniques to suggest similar tracks by genre, tempo, duration, and more.

Dataset Source: This project uses a song dataset introduced from this Kaggle notebook, which provides cleaned and normalized Spotify music data.

# Features
User registration and login with input validation
Song recommendation based on input song and optional filters (genre, year, tempo, duration)
Frontend built with HTML + JavaScript and styled for a smooth user experience
Backend powered by Flask with pandas and scikit-learn for data processing
SQLite database support (music.db) and CSV-based dataset (songs_normalize.csv)

# Project Structure
├── main.py               # Flask app and backend logic
├── index.html            # Frontend UI (rendered through Flask)
├── music.db              # SQLite database of users
├── songs_normalize.csv   # Dataset of songs for recommendation
├── pyproject.toml        # Python dependency manager
├── replit.nix            # Environment configuration for Replit
└── uv.lock               # Package lock file

# Setup Instructions
Requirements
Python 3.11+
Flask
pandas
scikit-learn

# Install Dependencies
pip install -r requirements.txt

# Run the App
python main.py

# Usage
Register as a new user.
Log in with your credentials.
Enter a song name and optional filters (genre, year, tempo, duration).
Receive a list of recommended songs from the dataset.

# Tech Stack
Backend: Python, Flask
Frontend: HTML, JavaScript
Data: pandas, scikit-learn
Database: SQLite
Deployment-Friendly: Replit + replit.nix

# Notes
Passwords are stored in plaintext (⚠️ not secure for production).
Recommendations are based on basic similarity logic—ideal for educational/demo purposes.
Make sure to set a proper app.secret_key for session handling.

# License
MIT License (or add your license of choice)

