Movie Recommendation System

A movie recommendation system developed entirely from scratch in Python, implementing multiple recommendation approaches without relying on pre-built recommendation-system libraries.

The project currently includes:

Item-Based Collaborative Filtering: Implemented from scratch, using similarities between movies based on users' ratings.

TF-IDF Content-Based Filtering: Implemented from scratch to represent movies using their textual metadata and calculate content similarities.

Word2Vec Content-Based Filtering: A custom Skip-Gram Word2Vec implementation, trained on movie descriptions to generate movie embeddings.

Pseudo-Ratings Generation: A custom approach that estimates ratings for movies with no or very few user ratings by leveraging similarities with other movies.

The system combines movie metadata from TMDB with user ratings from MovieLens 32M, enabling both content-based and collaborative recommendation approaches.

How to Run

The program currently runs only through the terminal. 

1. Clone or download the repository.

2. Open a terminal in the project's root directory.

3.   Run the following command:

python main.py

Dataset Generation

If you want to generate the dataset yourself, you will need your own TMDB API key. Add your API key to keys.py before running the dataset generation process.
