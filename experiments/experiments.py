import numpy as np
import pandas as pd
import pickle
import os
from cf import create_ratings_file,create_similarities_file,predict_rating,predict_rating_k
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

def create_split_ratings_files():
    split_path = os.path.join(EXPERIMENTS_DIR, "ratings_split.pkl")
    if os.path.exists(split_path): #if file already exists, return

        return

    with open("ratings.pkl", "rb") as f:  # get user_ratings from ratings.pkl
        movie_voters, user_ratings, user_means = pickle.load(f)
        del movie_voters
        del user_means

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

    del user_ratings #no use after this point, free RAM

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



def check_split_ratings_overlap(train_user_ratings, validation_user_ratings, test_user_ratings): #function checks if a rating (user_id,movie_index) exists in more than one set

    # user_ratings has shape: {key: user_id, value: {movie_index:rating}

    for user_id, ratings in train_user_ratings.items():

        if user_id in validation_user_ratings: #if user exists in both train set and validation set

            if ratings.keys() & validation_user_ratings[user_id].keys(): #ratings.keys() for the movie_indices in train set, validation_user_ratings[user_id].keys() for the movie_indices in validation set
                print("\nError, Train-Validation Overlap Found")
                return

        if user_id in test_user_ratings: #if user exists in both train set and test set

            if ratings.keys() & test_user_ratings[user_id].keys(): #ratings.keys() for the movie_indices in train set, test_user_ratings[user_id].keys() for the movie_indices in test set
                print("\nError, Train-Test Overlap Found")
                return

    for user_id, ratings in validation_user_ratings.items():

        if user_id in test_user_ratings: #if user exists in both validation set and test set

            if ratings.keys() & test_user_ratings[user_id].keys(): #ratings.keys() for the movie_indices in validation set, test_user_ratings[user_id].keys() for the movie_indices in test set
                print("\nError, Validation-Test Overlap Found")
                return

    print("\nNo Overlapping Found")
    return


def check_cold_start_movies(train_movie_voters, validation_movie_voters, test_movie_voters): #function checks the number of movies appearing in validation or test set, without being in train set


    # movie_voters has shape: {key: movie_index, value: {user_id:rating}

    # get about movie_index

    train_movies=train_movie_voters.keys()
    validation_movies=validation_movie_voters.keys()
    test_movies=test_movie_voters.keys()

    validation_test_movies=validation_movies | test_movies

    cold_start_movies=validation_test_movies-train_movies

    n_cold_start_movies=len(cold_start_movies)

    n_test_cold_start_reviews=(
        sum(len(test_movie_voters[movie_index])
            for movie_index in cold_start_movies
            if movie_index in test_movie_voters))

    n_validation_cold_start_reviews=(
        sum(len(validation_movie_voters[movie_index])
            for movie_index in cold_start_movies
            if movie_index in validation_movie_voters))

    n_cold_start_reviews=n_test_cold_start_reviews+n_validation_cold_start_reviews

    total_movies=len(validation_test_movies)

    n_validation_reviews=sum(len(voters) for voters in validation_movie_voters.values())

    n_test_reviews=sum(len(voters) for voters in test_movie_voters.values())

    total_reviews=n_validation_reviews+n_test_reviews

    percentage_cold_start_movies=n_cold_start_movies/total_movies*100

    percentage_cold_start_reviews=n_cold_start_reviews/total_reviews*100

    print(f"\nMovies appearing in validation or test set, but not in train set (Cold start movies) : {n_cold_start_movies}")

    print(f"\nCold start movie percentage: {percentage_cold_start_movies:.2f}%")

    print(f"\nCold start ratings: {n_cold_start_reviews}")

    print(f"\nCold start ratings percentage: {percentage_cold_start_reviews:.6f}%")

    return cold_start_movies





def check_split_ratings_details():

    create_json()

    df = pd.read_json("tmdb_dataset.json")  # Read json file

    create_ratings_file(df)  # use cf.py file

    # read file and load data
    create_split_ratings_files()

    with open(os.path.join(EXPERIMENTS_DIR, "ratings_split.pkl"), "rb") as f:
        train_user_ratings, train_movie_voters, train_user_means, validation_movie_voters, validation_user_ratings, test_user_ratings, test_movie_voters = pickle.load(f)

    train_count=sum(len(ratings) for ratings in train_user_ratings.values())
    validation_count=sum(len(ratings) for ratings in validation_user_ratings.values())
    test_count=sum(len(ratings) for ratings in test_user_ratings.values())

    total_count=train_count+validation_count+test_count


    print("\nDataset statistics:\n")

    print(f"Train Set:")
    print(f"  Users: {len(train_user_ratings):,}")
    print(f"  Movies: {len(train_movie_voters):,}")
    print(f"  Ratings: {train_count:,}")

    print(f"\nValidation Set:")
    print(f"  Users: {len(validation_user_ratings):,}")
    print(f"  Movies: {len(validation_movie_voters):,}")
    print(f"  Ratings: { validation_count:,}")

    print(f"\nTest Set:")
    print(f"  Users: {len(test_user_ratings):,}")
    print(f"  Movies: {len(test_movie_voters):,}")
    print(f"  Ratings: {test_count:,}")

    #Calculate percentages
    train_percentage=train_count/total_count*100
    validation_percentage=validation_count/total_count*100
    test_percentage=test_count/total_count*100

    print("\nRatings Distribution:")
    print(f"  Train Set:       {train_percentage:.2f}%")
    print(f"  Validation Set:  {validation_percentage:.2f}%")
    print(f"  Test Set:        {test_percentage:.2f}%")

    check_split_ratings_overlap(train_user_ratings, validation_user_ratings, test_user_ratings)
    cold_start_movies=check_cold_start_movies(train_movie_voters, validation_movie_voters, test_movie_voters)
    check_movies_with_reviews()

    return cold_start_movies


def check_movies_with_reviews():

    create_json()
    df = pd.read_json("tmdb_dataset.json")
    with open("ratings.pkl", "rb") as f:  # get movie_voters from ratings.pkl
        movie_voters, user_ratings, user_means = pickle.load(f)
        del user_ratings
        del user_means

    movies_with_reviews=len(movie_voters)

    print(f"\nMovies with reviews : {movies_with_reviews:,}")

    percentage_movies_with_reviews=movies_with_reviews/len(df)*100

    print(f"\nMovies with reviews percentage: {percentage_movies_with_reviews:.2f}%")

def create_train_set_similarities_file():

    directory = "experiments"
    file_name = "train_set_similarities.pkl"

    file_path = os.path.join(directory, file_name)

    if os.path.exists(file_path):
        return

    with open(os.path.join(EXPERIMENTS_DIR, "ratings_split.pkl"), "rb") as f:
        train_user_ratings, train_movie_voters, train_user_means, validation_movie_voters, validation_user_ratings, test_user_ratings, test_movie_voters = pickle.load(f)

    target_movies=validation_movie_voters.keys() & train_movie_voters.keys() #We want to calculate similarities only for movies appearing in both sets

    target_movie_voters={movie_index: train_movie_voters[movie_index]
                         for movie_index in target_movies}

    del train_movie_voters,validation_movie_voters, validation_user_ratings, test_user_ratings, test_movie_voters #no use, free RAM

    create_similarities_file(target_movie_voters,train_user_ratings,train_user_means,10,True)

def evaluate_cf_model(cold_start_movies): #function that evaluates cf_model, using MAE and RMSE, using different k's

    results_file=os.path.join(EXPERIMENTS_DIR, "cf_evaluation.txt") #create file with results

    if os.path.exists(results_file):
        print("Check cf_evaluation.txt for results")
        return

    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Collaborative Filtering Evaluation\n")
        f.write("=" * 45 + "\n")

    #use MAE and RMSE for evaluation

    directory = "experiments"

    # Load training and validation data
    with open(os.path.join(directory, "ratings_split.pkl"), "rb") as f:

        (train_user_ratings,train_movie_voters,train_user_means,validation_movie_voters,validation_user_ratings,test_user_ratings,test_movie_voters) = pickle.load(f)

        del validation_movie_voters,test_user_ratings,test_movie_voters

    # Load similarities
    with open(os.path.join(directory, "train_set_similarities.pkl"), "rb") as f:
        similarities = pickle.load(f)


    movies_means={movie_index : np.mean(list(movie_voters.values())) #Baseline for to compare predictions with (mean of movies ratings)
                  for movie_index,movie_voters in train_movie_voters.items()}

    k_values=[None,1,3,5,7,9,11,13,15,17,19,21,31,51,71,91]

    first_iteration = True #Calculate coverage and baseline errors, only on first iteration since they stay the same

    for k in k_values:

        absolute_errors = []
        squared_errors = []

        if first_iteration:
            baseline_absolute_errors = []
            baseline_squared_errors = []

            successful_predictions = 0
            total_validation_ratings = 0

        for user_id,ratings in validation_user_ratings.items(): #for each user and rating in validation set

            for movie_index,rating in ratings.items():

                if movie_index not in cold_start_movies:

                    if first_iteration:

                        total_validation_ratings += 1

                    if k is not None:

                        prediction=predict_rating_k(user_id,movie_index,train_user_ratings,train_user_means,similarities,k)

                    else:
                        prediction=predict_rating(user_id,movie_index,train_user_ratings,train_user_means,similarities)

                    if prediction is not None:

                        if first_iteration:

                            baseline_prediction = movies_means[movie_index]

                            baseline_error = rating - baseline_prediction

                            baseline_absolute_errors.append(abs(baseline_error))

                            baseline_squared_errors.append(np.square(baseline_error))

                            successful_predictions += 1

                        error = rating-prediction

                        absolute_errors.append(abs(error))

                        squared_errors.append(np.square(error))

        MAE=np.mean(absolute_errors)

        RMSE=np.sqrt(np.mean(squared_errors))

        if first_iteration:

            baseline_MAE=np.mean(baseline_absolute_errors)

            baseline_RMSE=np.sqrt(np.mean(baseline_squared_errors))

            coverage=successful_predictions/total_validation_ratings*100

        if k is not None:
            title=f"Collaborative Filtering Evaluation, Using k:{k}"

        else:
            title="Collaborative Filtering Evaluation (Without K)"

        if first_iteration:
            result =( f"\n{title}\n"
                f"RMSE: {RMSE:.4f}\n"
                f"Baseline RMSE: {baseline_RMSE:.4f}\n"
                f"MAE: {MAE:.4f}\n"
                f"Baseline MAE: {baseline_MAE:.4f}\n"
                f"Coverage: {coverage:.4f}%\n"
                + "-" * 45 + "\n")
        else:
            result = (f"\n{title}\n"
                      f"RMSE: {RMSE:.4f}\n"
                      f"MAE: {MAE:.4f}\n"
                      + "-" * 45 + "\n")

        first_iteration=False

        print(result)
        with open(results_file, "a", encoding="utf-8") as f:
            f.write(result)


if __name__=="__main__":

    cold_start_movies=check_split_ratings_details()

    create_train_set_similarities_file()

    evaluate_cf_model(cold_start_movies)








