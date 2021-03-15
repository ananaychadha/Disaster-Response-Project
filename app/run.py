import pandas as pd
import sys
import pandas as pd
import json
import plotly
import sqlalchemy
from sqlalchemy import create_engine
import nltk
import operator
import pickle
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request, jsonify
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline,FeatureUnion
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin


class PopularWords(BaseEstimator, TransformerMixin):
    def __init__(self, word_dict, pct = .001):
        
        self.word_dict = word_dict
        if pct == None:
            self.n = int(len(self.word_dict) * .01)
        else:
            self.n = int(len(self.word_dict) * pct)
        
    def fit(self, X, y = None):
       
        return self
            
    
    def transform(self, X):
       
        def get_word_count(message_list, top_word_count, sorted_dict):
           
           
            total_count = 0
            for w in range(top_word_count):
                if sorted_dict[w][0] in message_list:
                    total_count +=1
        
            return total_count 
        
        # Sort the dictionary from most frequent words to least frequent words
        sorted_dict = sorted(self.word_dict.items(), key=operator.itemgetter(1), reverse = True)
        
        # Make the words lowercase
        lower_list = pd.Series(X).apply(lambda x: x.lower())
        
        # Get rid of punctuation
        no_punct = pd.Series(lower_list).apply(lambda x: re.sub(r'[^\w\s]','', x))
        
        # Create list of the words that are not stop words
        final_trans = pd.Series(no_punct).apply(lambda x: x.split())
        
        # Get the top number of words that want to be viewed from the dictionary
        top_word_cnt = self.n
        
        # Get the results
        results = pd.Series(final_trans).apply(lambda x: get_word_count(x,top_word_cnt, sorted_dict)).values
        # Put in DataFrame output to work with pipeline
        return pd.DataFrame(results)


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #genre and aid_related status
    related_aid = df[df['aid_related']==1].groupby('genre').count()['message']
    related_aid1 = df[df['aid_related']==0].groupby('genre').count()['message']
    genre_names = list(related_aid.index)

    # let's calculate distribution of classes with 1
    distribution_of_class = df.drop(['id', 'message', 'original', 'genre'], axis = 1).sum()/len(df)

    #sorting values in ascending
    distribution_of_class = distribution_of_class.sort_values(ascending = False)

    #series of values that have 0 in classes
    distribution_of_class1 = (distribution_of_class -1) * -1
    class_name = list(distribution_of_class.index)


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=related_aid,
                    name = 'Aid is related'

                ),
                Bar(
                    x=genre_names,
                    y= related_aid1,
                    name = 'Aid is not related'
                )
            ],

            'layout': {
                'title': 'Distribution of message by genre and \'aid related\' class ',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'barmode' : 'group'
            }
        },
        {
            'data': [
                Bar(
                    x=class_name,
                    y=distribution_of_class,
                    name = 'Class = 1'
                    #orientation = 'h'
                ),
                Bar(
                    x=class_name,
                    y=distribution_of_class1,
                    name = 'Class = 0',
                    marker = dict(
                            color = 'rgb(212, 228, 247)'
                                )
                    #orientation = 'h'
                )
            ],

            'layout': {
                'title': 'Distribution of labels within classes',
                'yaxis': {
                    'title': "Distribution"
                },
                'xaxis': {
                    'title': "Class",
            #        'tickangle': -45
                },
                'barmode' : 'stack'
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
