import graphlab
song_data=graphlab.SFrame('song_data.sframe') #loading the dataset

#spliting the data into traing and test set
train_data,test_data=song_data.random_split(.8,seed=0)

#unique users
users=song_data['user_id'].unique()

#building a personalized model
personalized_model=graphlab.item_similarity_recommender.create(train_data,user_id='user_id',item_id='song')

#output of the personalized model
personalized_model.recommend(users=[users[0]])
personalized_model.recommend(users=[users[1]])

#if user likes a song A then what all other songs he may like
personalized_model.get_similar_items(['Naked - Marques Houston'])

#to get the count of the unique listiners of the artist Kanye West
west=song_data[song_data['artist']=='Kanye West']
west_users=west['user_id'].unique()
len(west_users)
#to get the count of how many listerns have listened to the songs of different users
a=song_data.groupby(key_columns='artist',operations={'total_count':graphlab.aggregate.SUM('listen_count')})

# most recommended songs
subset_test_users = test_data['user_id'].unique()[0:10000]

b=personalized_model.recommend(subset_test_users,k=1)
c=b.groupby(key_columns='song',operations={'count':graphlab.aggregate.COUNT()})
c=c.sort('count',ascending=False)
c
