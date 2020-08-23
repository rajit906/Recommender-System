# Recommender-System

### Setting up Data Base
Create a Data Folder and upload `sample-data.csv`

### Cleaning the dataset
Run preprocess.py with `python preprocess.py`

Create a models folder

### Creating Word2Vec Model
Run embeddings.py with `python embeddings.py`

This loads a pkl model

### Running the flask app
Finally run app.py with `python app.py`

Interact with the API using: 
    #GET REQUEST - http://127.0.0.1:5000/get_recos?user_desc=%22green%20shirt%22&n=2

    #POST REQUEST - curl -X POST -d user_desc="hat" -d n=3 127.0.0.1:5000/get_recos 
