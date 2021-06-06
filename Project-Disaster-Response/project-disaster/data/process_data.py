import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT - messages_filepath - string 
            categories_filepath -string
    OUTPUT - 
            df- pandas dataframe
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories)
    return df
    


def clean_data(df):
    '''
    splits classified categories and removes categories columns with all zeros
    INPUT - df- pandas dataframe
    OUTPUT - 
            df- pandas dataframe
    '''
    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x[-1])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    not_binary_cols=categories.nunique().index[categories.nunique() == 1].tolist()
    categories.drop(not_binary_cols,axis=1,inplace=True)
    df.drop(['categories'],axis=1,inplace=True)
    df=pd.concat([df,categories],axis=1)
    df.drop_duplicates(['id'],inplace=True)
    mode=df['related'].mode()[0]
    df['related']=df['related'].apply(lambda x:mode if x not in (0,1) else x)
    return df


def save_data(df, database_filename):
    '''
    saves table to a fiel
    INPUT - df- pandas dataframe
            database_filename - string
    OUTPUT -
    '''
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DISASTER_MERGED', engine, index=False,if_exists='replace')  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()