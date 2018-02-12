from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk, re
from nltk.stem.wordnet import WordNetLemmatizer
import pprint
import json
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
import pymongo
from pymongo import MongoClient

def ConnectToMongo (connectionString):
    try:
        print('Connecting to Mongo Client -', connectionString, '....')
        client = MongoClient(connectionString)
        db = client.sampleDB
        collection = db.myColl
        print('Connection to Mongo Client -', connectionString, 'Successful...')
        print('Collection Details : ', collection)
    except Exception as e:
        print ('Unable to establish Connection : ', e)
    return collection


num_titles = 20000
myinput = []

collection = ConnectToMongo('mongodb://127.0.0.1:27017')

print("Begin Fetching data from MongoDB ... ")
listOfTitles = list(collection.find({}, {"title": 1, "_id": 0}).limit(num_titles))
print("Done Fetching ", num_titles ," data from MongoDB ... ")


# listOfTitles = list(collection.aggregate(
#     [
#         {'$sample': {'size': num_titles }},
#         {'$project' : { "title" : 1, "_id":0}}
#     ]
# ))

for eachTitleFromListOfTitles in listOfTitles:
    myinput.append(eachTitleFromListOfTitles['title'])

#print("\n\nmyinput : ", myinput)


vectorizer = CountVectorizer(min_df=1, max_df=0.95,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}',
                             ngram_range=(1, 3))
data_vectorized = vectorizer.fit_transform(myinput)

# Build a Latent Dirichlet Allocation Model
lda_model = LatentDirichletAllocation(n_components=4, max_iter=50, learning_method='online', random_state=0)
lda_Z = lda_model.fit_transform(data_vectorized, num_titles)

print("\n\nNO_DOCUMENTS, NO_TOPICS (n_components) : ", lda_Z.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in different topic spaces
print( "\n\nlda_z : ",  lda_Z[0])

model = (vectorizer, data_vectorized, lda_model.components_ , lda_model.exp_dirichlet_component_,lda_model.doc_topic_prior_)

print("Start pickling LDA Model")
import pickle
pickle.dump(model, open( "LDAModel_Pickle.p", "wb" ))
print("Done pickling LDA Model")