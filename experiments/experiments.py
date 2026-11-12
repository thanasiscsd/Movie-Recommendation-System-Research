import pandas as pd
import numpy as np
import pickle
import os


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

    with open("")







