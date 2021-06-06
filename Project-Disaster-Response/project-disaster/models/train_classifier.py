import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
import nltk
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import pickle


from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from utils.token_fn import tokenize

nltk.download(['punkt','stopwords','wordnet'])
stemmer=PorterStemmer()
lemmatizer=WordNetLemmatizer()
url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def load_data(database_filepath):
    '''
    Reads table from db file
    INPUT - database_filepath- string
    OUTPUT - 
            X - pandas Series of messages
            Y - pandas data frame of target columns 
            Z - list of target columns
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('select * from DISASTER_MERGED',engine)
                           
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    return X,Y,Y.columns.tolist()




def build_model():
    '''
    Defines and returns sklearn pipeline
    OUTPUT - 
            pipeline - sklearn pipeline
          
    '''
    pipeline = Pipeline(
    [
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(AdaBoostClassifier(DecisionTreeClassifier(max_depth=1,class_weight='balanced'),n_estimators=100)))   
    ]
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Reads table from db file
    INPUT - model- sklearn model
            X_test - pandas Series of messages
            Y_test - pandas data frame of target columns 
            category_names - list of target columns
    '''
    pipeline=model
    predict_classes=pipeline.predict(X_test.values)
    for i in range(Y_test.values.shape[1]):
        actual_classes=Y_test.values[:,i]
        pred_classes=predict_classes[:,i]
        print(classification_report(actual_classes,pred_classes,labels=[1],target_names=[category_names[i]]) )

def save_model(model, model_filepath):
    '''
    serializes mode
    INPUT - model- sklearn model
            model_filepath - string
    '''
    pickle.dump(model,open(model_filepath,'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()