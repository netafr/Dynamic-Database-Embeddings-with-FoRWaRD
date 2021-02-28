import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ARGVA
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec

import argparse
import numpy as np
import networkx as nx
import torch.utils.data
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold

from db_utils import Database
from numpy import mean, std
import random
import io_utils
import glob
import os
import re

import node2vec
from io_utils import save_dict
from shutil import copyfile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_node2vec_embedding_new(G, epochs=5, channels=100):
    embedding, model = node2vec.node2vec_embedding(G, 40, 30, 5, embed_dim=channels, neg_samples=20, batch_size=40000,
                                                   epochs=epochs)
    embedding = {n: embedding[i] for i, n in enumerate(G.nodes())}
    return embedding, model


def compute_embedding(db):
    G = db.get_row_val_graph()
    embedding, model = get_node2vec_embedding_new(G, epochs=5)
    return embedding, model



def dynamic_neighbors_embedding(new_db, embedding, feature_size=100):
    row_nodes = [row_id for _, row_id, _ in new_db.iter_rows()]
    for rel_id, row_id, row in new_db.iter_rows():
        count = 0
        curr_embedding = np.zeros(feature_size)
        for col, cell in row.items():
            col_id = f'{col}@{rel_id}'
            if not col_id == new_db.predict_col:
                values = cell.split() if type(cell) == str else [cell]
                for val in values:
                    val_id = f'{val}@{col_id}'
                    if val_id in embedding:
                        curr_embedding = np.add(curr_embedding, embedding[val_id])
                    else:
                        curr_embedding = np.add(curr_embedding, np.float32(np.random.normal(0.0, 1.0, feature_size)))
                    count += 1
        embedding[row_id] = np.true_divide(curr_embedding, count) if count > 0 else curr_embedding
    return embedding


def dynamic_similar_tuples_embedding(new_db, embedding, old_db, feature_size=100):
    G_old = old_db.get_row_val_graph()
    G_new = new_db.get_row_val_graph()
    row_nodes = [row_id for _, row_id, _ in new_db.iter_rows()]
    for rel_id, row_id, row in new_db.iter_rows():
        neighbors = list(G_new[row_id])
        similar_tuples = [list(G_old[neighbor]) for neighbor in neighbors if neighbor in embedding]
        similar_tuples = [inner for outer in similar_tuples for inner in outer]  # unites to one big list
        similar_tuples = list(set(similar_tuples))  # unique
        similar_tuples_filtered = []
        count = 0
        for t in similar_tuples:
            t_neighbors = list(G_old[t])
            common_per = len(list(set(neighbors) & set(t_neighbors))) / len(neighbors)
            if common_per > 0.3:
                count += 1
                similar_tuples_filtered.append(t)
        similar_tuples_embedding = np.array([np.array(embedding[x]) for x in similar_tuples_filtered])
        embedding[row_id] = np.mean(similar_tuples_embedding, axis=0) if count > 0 else np.float32(
            np.random.normal(0.0, 1.0, feature_size))

    return embedding


def dynamic_similar_tuples_weighted_embedding(new_db, embedding, old_db, feature_size=20):
    G_old = old_db.get_row_val_graph()
    G_new = new_db.get_row_val_graph()
    row_nodes = [row_id for _, row_id, _ in new_db.iter_rows()]
    for rel_id, row_id, row in new_db.iter_rows():
        neighbors = list(G_new[row_id])
        similar_tuples = [list(G_old[neighbor]) for neighbor in neighbors if neighbor in embedding]
        similar_tuples = [inner for outer in similar_tuples for inner in outer]  # unites to one big list
        similar_tuples = list(set(similar_tuples))  # unique
        similar_tuples_a = []
        count = 0
        for t in similar_tuples:
            t_neighbors = list(G_old[t])
            common_per = len(list(set(neighbors) & set(t_neighbors))) / len(neighbors)
            similar_tuples_a.append(common_per)

        similar_tuples_a /= np.sum(similar_tuples_a)
        similar_tuples_embedding = np.array(
            [np.array(embedding[x]) * a_i for x, a_i in zip(similar_tuples, similar_tuples_a)])
        embedding[row_id] = np.mean(similar_tuples_embedding, axis=0) if count > 0 else np.float32(
            np.random.normal(0.0, 1.0, feature_size))

    return embedding



def dynamic_gradient_embedding(db):
    G = db.get_row_val_graph()
    G_static = db.get_row_val_graph(partition=0)
    embedding, model = node2vec.node2vec_dynamic_embedding(G, G_static, 40, 30, 5, embed_dim=100, neg_samples=20,
                                                           batch_size=40000)
    embedding = {n: embedding[i] for i, n in enumerate(G.nodes())}
    return embedding, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ration of train data")
    args = parser.parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    name = 'mondial'  # choices: genes, carcinogenesis, hepatitis, medical
    embedding_name = 'test_d1.pckl'

    model_dir = f'models/n2v/dynamic/mondial/{args.train_ratio}'
    os.makedirs(model_dir, exist_ok=True)

    path = f'Datasets/{name}'
    embedding_path = f'Embeddings/{name}/dynmaic_{embedding_name}'

    # create temp files splitted
    relation_paths = glob.glob(os.path.join(path, '*.csv'))
    relations = {os.path.basename(p).split('.')[0]: pd.read_csv(p) for p in relation_paths}

    # duplicate the cols spec
    col_spec_path = glob.glob(os.path.join(path, '*cols'))[0]
    copyfile(col_spec_path, path + '/static/' + os.path.basename(col_spec_path))
    copyfile(col_spec_path, path + '/dynamic/' + os.path.basename(col_spec_path))

    db = Database.load_csv(path)
    Y, rows = db.get_labels()

    ratio = args.train_ratio
    print(str(ratio) + "!!!")

    scores = []
    split = StratifiedShuffleSplit(train_size=ratio, random_state=0, n_splits=10)
    for i, (train_index, test_index) in enumerate(split.split(rows, Y)):
        train_rows = [rows[j] for j in train_index]
        test_rows = [rows[j] for j in test_index]
        partition = {**{t: 0 for t in train_rows}, **{t: 1 for t in test_rows}}
        db.partition(partition=partition)

        embedding, _ = dynamic_gradient_embedding(db)
        # embedding, model = io_utils.load_or_compute(embedding_path, lambda: compute_embedding(db_static), use=False)
        # embedding = dynamic_similar_tuples_embedding(db_dynamic, embedding, db_static)

        X_train = np.float32([embedding[rows[j]] for j in train_index])
        X_test = np.float32([embedding[rows[j]] for j in test_index])
        Y_train, Y_test = [Y[j] for j in train_index], [Y[j] for j in test_index]

        model = SVC(random_state=1, max_iter=300)
        clf = make_pipeline(StandardScaler(), model)
        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)
        print(f'Ratio {ratio}, Run {i}: {score}')
        save_dict({'scores': scores}, f'{model_dir}/results.json')
        scores.append(score)

    print(f"Ratio {ratio} Result: + {np.mean(scores)} (+-{np.std(scores)})")
    print("--------")