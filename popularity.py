import graphlab
song_data=graphlab.SFrame('song_data.sframe') #loading the dataset

#spliting the data into traing and test set
train_data,test_data=song_data.random_split(.8,seed=0)

#unique users
users=song_data['user_id'].unique()

#building a model based on the popularity of the song
popularity_model=graphlab.popularity_recommender.create(train_data,user_id='user_id',item_id='song')

#output of the popularity model
popularity_model.recommend(users=[users[0]])
popularity_model.recommend(users=[users[1]])
