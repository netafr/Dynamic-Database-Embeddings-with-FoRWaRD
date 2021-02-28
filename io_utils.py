import pandas as pd
import os
import pickle
import json


def load_embedding(path):
    with open(path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding


def save_embedding(path, embedding):
    if not os.path.exists(os.path.dirname(path)):
        os.mkdir(os.path.dirname(path))

    with open(path, 'wb') as f:
        pickle.dump(embedding, f)


def load_or_compute(path, embedding_fct):
    if not os.path.exists(path):
        embedding = embedding_fct()
        save_embedding(path, embedding)
    else:
        embedding = load_embedding(path)

    return embedding


def save_dict(dict, path):
    with open(path, 'w') as f:
        json.dump(dict, f, indent=4)