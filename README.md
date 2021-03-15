Disaster-Response-Project Repository for the course project Disaster Respose as part of the Nanodegree Data Science from Udacity.

Project Components This repository has 3 components:

ETL Pipeline (process_data.py) Loads the messages and categories datasets Merges the two datasets Cleans the data Stores it in a SQLite database

ML Pipeline (train_classifier.py) Loads data from the SQLite database Splits the dataset into training and test sets Builds a text processing and machine learning pipeline Trains and tunes a model using GridSearchCV Outputs results on the test set Exports the final model as a pickle file

Flask Web App (app folder) Provides basic descriptives of the training data Has a form to classifify new messages using the best ML model

Install This project requires Python 3.x and the following Python libraries installed:

NumPy Pandas Matplotlib Json Plotly Nltk Flask Sklearn Sqlalchemy Sys Re Pickle

Usage: In the project's root directory:

To run ETL pipeline that cleans data and stores in database: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/InsertDatabaseName.db

To run ML pipeline that trains classifier and saves: python models/train_classifier.py data/InsertDatabaseName.db models/classifier.pkl

To run the web app locally: python app/run.py then go to http://0.0.0.0:3001/ or localhost:3001

Alternatevely, in unix system type: gunicorn app.run:app -b 0.0.0.0:3001 to run a local gunicorn server

I couldn't upload the findings (screenshots) on github for some reason, so I made a word document

Here's the link for that. https://docs.google.com/document/d/1WrwBdI2lpag5fcR8bbJNL_gC-JuKk9Iy5Hhby1uxq3g/edit?usp=sharing
