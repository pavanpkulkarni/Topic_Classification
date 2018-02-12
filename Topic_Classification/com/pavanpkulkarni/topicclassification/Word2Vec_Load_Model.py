from gensim.models import KeyedVectors

model_path = '/Users/pavanpkulkarni/OneDrive - Harrisburg University/Fall_2017/Analysis_of_the_Human_Language/Final_Project/'


def load_wordvec_model(modelName, modelFile, flagBin):
    print('Loading ' + modelName + ' model...')
    model = KeyedVectors.load_word2vec_format(model_path + modelFile, binary=flagBin)
    print('Finished loading ' + modelName + ' model...')
    return model

model_word2vec = load_wordvec_model('Word2Vec', 'GoogleNews-vectors-negative300.bin', True)

# import _pickle as cPickle
# cPickle.dump(model_word2vec, open('Word2Vec_Pickle.p','wb'))

model_word2vec.save('Word2Vec_Pickle.p')