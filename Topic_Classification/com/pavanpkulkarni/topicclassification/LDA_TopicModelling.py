
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# create a blank model
lda = LatentDirichletAllocation()

print("Loading LDA model from pickle..")
(vectorizer, data_vectorized,lda.components_, lda.exp_dirichlet_component_,lda.doc_topic_prior_) = pickle.load( open( "LDAModel_Pickle.p", "rb" ) )

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d : " % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]])


# components_: array, [n_components, n_features]
# Variational parameters for topic word distribution. Since the complete conditional for topic word distribution is a Dirichlet, components_[i, j] can be viewed as pseudocount that represents the number of times word j was assigned to topic i. It can also be viewed as distribution over the words for each topic after normalization
# (http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.LatentDirichletAllocation.html)


print("\n\n LDA Model (Word - Topic  Distribution) : ")
print("=" * 20)
print_topics(lda, vectorizer)
print("=" * 20)

print("\n\n Topic - Document Distribution :  ")

text = ["Apple does not raise prices for black friday deals",
        "Mail carriers accuse USPS of faking Amazon delivery records so",
        "Taking on Amazon and Google turns out to be harder than Apple"]

tf_vectorizer = CountVectorizer(vocabulary=vectorizer.get_feature_names())
tf = tf_vectorizer.fit_transform(text)

predict = lda.transform(tf)
print("\n\n LDA Prediction : ", predict)
