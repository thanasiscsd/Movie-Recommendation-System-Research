import pandas as pd

from surprise import Dataset, Reader, SVD, accuracy


def train_matrix_factorization(train_user_ratings, test_user_ratings):

    # Convert training ratings to DataFrame
    train_data = [
        (user_id, movie_id, rating)
        for user_id, ratings in train_user_ratings.items()
        for movie_id, rating in ratings.items()
    ]

    train_df = pd.DataFrame(train_data,columns=["user_id", "movie_id", "rating"]) #convert to pd Dataframe

    reader = Reader(rating_scale=(0.5, 5.0)) #minimum rating is 0.5, maximum is 5.0

    data = Dataset.load_from_df(train_df[["user_id", "movie_id", "rating"]],reader) #convert to surprise from pd

    trainset = data.build_full_trainset()

    # Initialize SVD
    model = SVD(n_factors=100,n_epochs=20,lr_all=0.005,reg_all=0.02,random_state=25)

    # Train
    model.fit(trainset)

    del train_data, train_df, data #no use anymore, free RAM

    # Prepare test set
    testset = [
        (str(user_id), str(movie_id), float(rating))
        for user_id, ratings in test_user_ratings.items()
        for movie_id, rating in ratings.items()
    ]

    predictions = model.test(testset)

    print("RMSE:", accuracy.rmse(predictions))
    print("MAE:", accuracy.mae(predictions))

    return model