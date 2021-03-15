import sys
import pandas as pd
import sqlalchemy
import numpy as np
from sqlalchemy import create_engine

# Load in the messages and the affiliated categories
def load_data(messages_filepath, categories_filepath):

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the two datasets together
    df = pd.merge(messages, categories)

    return df


def clean_data(df):
    """
    Cleaning data and dropping duplicates on the disaster response dataset
    """
    
    

    # Creating a dataframe with the 36 individual category columns
    categories = df['categories'].str.split(pat = ';', expand = True)

    #Choosing the first row from the categories df
    row = categories.iloc[0]

    # Strip the value before the "-"
    category_colnames = [val.split('-')[0] for val in row.values]

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(np.int)
        

    # Drop the old categories columns
    df.drop('categories', axis= 1, inplace =True)

    # Add the original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1, join_axes=[df.index])
    df.related.replace(2,1,inplace=True)
    df.drop_duplicates(inplace = True)

    return df


def save_data(df, database_filename, table_name = 'InsertTableName'):

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('InsertTableName', engine, index=False)


def main():
    """
    Run ETL of messages and categories data
    """
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
