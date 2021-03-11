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

def tokenize(text, url_place_holder_string="urlplaceholder"):
    
    '''
    Tokenize the input string
    
    Input:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the input text message
    '''
    
    # Replace all urls with a default urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # find all urls from the input text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace the url with the default url placeholder string
    for url in detected_urls:
        text = text.replace(url, url_place_holder_string)

    # Extract word tokens from the input text
    tokens = word_tokenize(text)
    
    #Lemmatize to remove different inflected forms of a word so they can be analysed as a single item
    final_tokens = [WordNetLemmatizer().lemmatize(w).lower().strip() for w in tokens]
    return final_tokens

#Build a custom transformer to extract the starting verb of a sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    Starting Verb Extractor class    
    It extract the starting verb of a sentence and creates a new feature for the ML classifier
    
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    # Given it is a tranformer we can return the self 
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    """
    Build a Machine learning pipeline
    
    Output:
        A Machine Learning Pipeline that process input text messages and apply a classifier.
        
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_transformer', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline

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