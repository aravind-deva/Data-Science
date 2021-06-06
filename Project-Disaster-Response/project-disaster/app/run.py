import json
import plotly
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.getcwd())

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Pie
#from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

from utils.token_fn import tokenize
#print(tokenize)

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
engine = create_engine('sqlite:///./data/DisasterResponse.db')
df = pd.read_sql_table('DISASTER_MERGED', engine)
Y = df.drop(['id','message','original','genre'],axis=1)
# load model
model = joblib.load("./models/classifier.pkl")
urls=np.array(np.repeat('https://google.com',35),dtype='object')
urls[3:5]='https://www.redcross.org/'
urls[[5,6]]='https://www.americares.org/'
urls[9]='https://www.samaritanspurse.org/'
urls[[18,26]]='https://www.allhandsandhearts.org/'
urls[[6,11]]='https://www.salvationarmyusa.org/usn/'
urls[28]='http://sbpusa.org/'
urls[29]='https://teamrubiconusa.org/'
urls[31]='https://www.unicefusa.org/'


orgs=np.array(np.repeat('Search The Web',35),dtype='object')
orgs[3:5]='Red Cross'
orgs[[5,6]]='Ameri Cares'
orgs[9]='Samaritan'
orgs[[18,26]]='All Hands and Hearts'
orgs[[6,11]]='Salvation Army'
orgs[28]='SBP'
orgs[29]='Rubico'
orgs[31]='UNICEF'

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    #genre_counts = #df.groupby('genre').count()['message']
    class_counts=Y.melt().groupby('variable').sum()['value'].sort_values(ascending=False)[:6]
    class_names = list(class_counts.index)
    
    class_counts2=Y.melt().groupby('variable').sum()['value']
    class_names2 = list(class_counts2.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Pie(
                    labels=class_names,
                    values=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Top Message Categories'
            }
        },
         {
            'data': [
                Bar(
                    x=class_names2,
                    y=class_counts2
                )
            ],

            'layout': {
                'title': 'Distribution of All Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
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
    classification_results = dict(zip( zip(df.columns[4:],urls,orgs), classification_labels))

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