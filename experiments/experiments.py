import numpy as np
import pandas as pd
import pickle
import os
from cf import create_ratings_file
from main import create_json

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_DIR)
EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))

# movie_voters has shape: {key: movie_index, value: {user_id:rating}
# user_ratings has shape: {key: user_id, value: {movie_index:rating}

def create_movie_voters(user_ratings):
    movie_voters={}
    for user_id, ratings in user_ratings.items():  # for each user and his ratings

        for movie_index, rating in ratings.items(): #for each movie index and its rating

            if movie_index not in movie_voters: #if movie_index is not stored in dictionary
                movie_voters[movie_index]={} #create empty slot for current movie_index

            movie_voters[movie_index][user_id]=rating #add rating

    return movie_voters



# 80% training, 10% validation, 10% test

def create_split_ratings_files(user_ratings):
    split_path = os.path.join(EXPERIMENTS_DIR, "ratings_split.pkl")
    if os.path.exists(split_path): #if file already exists, return

        return

    seed=25 #use standard seed
    rng = np.random.default_rng(seed) #use seed for numpy

    train_user_ratings={}
    validation_user_ratings = {}
    test_user_ratings = {}

    #user_ratings has shape: {key: user_id, value: {movie_index:rating}

    for user_id,ratings in user_ratings.items(): #for each user and his ratings

        items=list(ratings.items()) #convert dictionary items to list

        rng.shuffle(items) #shuffle list values

        n=len(items) #number of votes

        #for users with less than 3 ratings, keep them only in training
        if n<3:
            train_user_ratings[user_id]=dict(items) #convert list back to dictionary
            continue

        n_validation=max(1,int(n*0.1)) #at least 1 review, else 10%
        n_test=max(1,int(n*0.1)) #at least 1 review, else 10%

        test_items=items[:n_test] #get from start until n_test

        validation_items=items[n_test:n_test + n_validation] #get from end of n_test (so test set and validation don't consist of the same data) until n_validation

        train_items=items[n_test + n_validation:] #get from n_test+n_validation until end of values

        #convert back to dictionary

        train_user_ratings[user_id]=dict(train_items)
        validation_user_ratings[user_id] = dict(validation_items)
        test_user_ratings[user_id] = dict(test_items)

    # Do the same for movie voters

    train_movie_voters=create_movie_voters(train_user_ratings)
    validation_movie_voters=create_movie_voters(validation_user_ratings)
    test_movie_voters=create_movie_voters(test_user_ratings)

    # Calculate user means using only the train set to avoid data leakage

    # Dictionary {key: user_id, value: average rating}

    train_user_means = {user_id: sum(train_user_ratings[user_id].values()) / len(train_user_ratings[user_id])
                  for user_id in train_user_ratings}

    all_data=(train_user_ratings, train_movie_voters, train_user_means,validation_movie_voters, validation_user_ratings, test_user_ratings, test_movie_voters)

    with open(split_path, "wb") as f:

        pickle.dump(all_data,f,protocol=pickle.HIGHEST_PROTOCOL)

    print("ratings_split.pkl successfully created")


create_json()

df = pd.read_json("tmdb_dataset.json") #Read json file

create_ratings_file(df) #use cf.py file

with open("ratings.pkl", "rb") as f: #get user_ratings from ratings.pkl
    _,user_ratings,_ = pickle.load(f)

#read file and load data
create_split_ratings_files(user_ratings)
with open(os.path.join(EXPERIMENTS_DIR, "ratings_split.pkl"), "rb") as f:
    train_user_ratings, train_movie_voters, train_user_means,validation_movie_voters, validation_user_ratings, test_user_ratings, test_movie_voters = pickle.load(f)

print("\nDataset statistics:")

print(f"Train:")
print(f"  Users: {len(train_user_ratings):,}")
print(f"  Movies: {len(train_movie_voters):,}")
print(f"  Ratings: {sum(len(ratings) for ratings in train_user_ratings.values()):,}")

print(f"\nValidation:")
print(f"  Users: {len(validation_user_ratings):,}")
print(f"  Movies: {len(validation_movie_voters):,}")
print(f"  Ratings: {sum(len(ratings) for ratings in validation_user_ratings.values()):,}")

print(f"\nTest:")
print(f"  Users: {len(test_user_ratings):,}")
print(f"  Movies: {len(test_movie_voters):,}")
print(f"  Ratings: {sum(len(ratings) for ratings in test_user_ratings.values()):,}")







