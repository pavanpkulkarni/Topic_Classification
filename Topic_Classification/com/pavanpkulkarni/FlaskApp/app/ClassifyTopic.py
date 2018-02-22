from flask import Flask, render_template, request
from ClassifyForm import ClassificationTitleForms

from flask import Flask
from flask import jsonify
from flask import request
from flask_pymongo import PyMongo
from bson import BSON
from bson import json_util
import json
from com.pavanpkulkarni.topicclassification.Word2Vec_TopicClassifier import classify_topics, topic_taxonomy

app = Flask(__name__)
app.secret_key = 'my secret key'

from gensim.models import KeyedVectors

import os
pickle_location = "/Users/pavanpkulkarni/OneDrive - Harrisburg University/Fall_2017/Analysis_of_the_Human_Language/Final_Project/Topic_Classification/com/pavanpkulkarni/topicclassification/Word2Vec_Pickle.p"

print("Loading Word2Vec model pickle..", pickle_location)
model_word2vec = KeyedVectors.load(pickle_location)

@app.route('/')
@app.route('/tclassify', methods=['POST', 'GET'])
def tclassify():

    form = ClassificationTitleForms()

    if request.method == 'POST':
        if form.submit.data:
            title = request.form['key']
            print("Title is : ", title)
            results = classify_topics(title, model_word2vec)
            return json.dumps(results, indent=4, default=json_util.default)
        elif form.topic_taxonomy.data:
            print("Topic Taxonomy is  : ", topic_taxonomy)
            return jsonify(**topic_taxonomy)

    elif request.method == 'GET':
        return render_template('classify.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)



# output1 = classify_topics('Amazon CEO and world’s richest man Jeff Bezos avoids a common', model_word2vec)
# print(output1)
#
# output2 = classify_topics('Researchers find "simple" way to hack Amazon Key', model_word2vec)
# print(output2)
#
# output3 = classify_topics('Mail carriers: USPS supervisors warn Amazon customers will get', model_word2vec)
# print(output3)
#
# output4 = classify_topics('Your New Mailman Works for Amazon', model_word2vec)
# print(output4)