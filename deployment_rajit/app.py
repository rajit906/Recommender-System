import flask
from flask import request, jsonify, render_template
from utils import get_similar_products, compute_user_input_embedding, preproc_user_input, serve_recos, remove_html
import pickle as pkl
import pandas as pd
import numpy as np

app = flask.Flask(__name__)
app.config['DEBUG'] = True

@app.route('/') #Home Page
def home():
    return "<h1>Ecommerce Product Recommender</h1><p>This site is a prototype API for serving recommendations in response to user item queries.</p>"

@app.route('/get_recos', methods = ['GET', 'POST']) # After inputting word, get directed to this
def get_recos():
    if request.method == "GET":
        #obtain user input
        user_desc = request.args['user_desc']
        num_recs = int(request.args['n'])

    elif request.method == 'POST':
        user_desc = request.form['user_desc']
        num_recs = int(request.form['n'])

    sim_prod = get_similar_products(compute_user_input_embedding(preproc_user_input(user_desc, model), model),catalog_embeddings, num_recs)    
    id_list = map(lambda tup: tup[0], sim_prod)
    recos = serve_recos(id_list, catalog)
    cleaned_recos = [remove_html(reco) for reco in recos]
    results = []
    for i in range(len(cleaned_recos)):
        d = {
            'rank': i + 1,
            'prod_desc': cleaned_recos[i]
        }

        results.append(d)

    return jsonify(results)

def embedding_to_list(txt):
    as_list = txt.replace("[", "").replace("]", "").split(",")
    return list(map(lambda s: float(s.strip()), as_list))

if __name__ == "__main__":
    MODEL_PATH = "models/w2v_model.pkl"
    DATA_PATH = "Data/sample-data.csv"
    EMBEDDINGS_PATH = "Data/ecomm_embeddings.csv"
    #load the model
    model = pkl.load(open(MODEL_PATH, "rb"))
    #load the raw catalog
    catalog = pd.read_csv(DATA_PATH)
    #load the catalog with embeddings
    catalog_embeddings = pd.read_csv(EMBEDDINGS_PATH)
    catalog_embeddings['desc_embedding'] = catalog_embeddings['desc_embedding'].apply(embedding_to_list)
    #Run the app
    app.run()











































