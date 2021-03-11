#import Libraries
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import pickle

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin
from scipy.stats import gmean

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath, table_name = 'Disaster_response'):
    
    """
    Load Data from the Database and Extract X and y variables for the modelling
    
    Inputs:
        database_filepath -> Path to SQLite destination database (In our case, it is: DisasterResponse.db)
    Output:
        X -> a dataframe containing the feature variables
        y -> a dataframe containing labels
        category_names -> List of category names
    """    
    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    
    df = pd.read_sql_table(table_name,engine)

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name,engine)
    
    #Lets drop the child_alone column since it has all zero values
    df = df.drop(['child_alone'],axis=1)
    
    # Since the related column has values either 1 or 2 and has 188 values stored as 2,
    #hence before we proceed lets replace these values with 1 to consder it as a valid response.
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    # Extract X and y variables from the data for the modelling
    X = df['message']
    y = df.iloc[:,4:]
    
    # Create category_names with the y column names for visualization purpose
    category_names = y.columns 
    
    return X, y, category_names


def tokenize(text):
    pass


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


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