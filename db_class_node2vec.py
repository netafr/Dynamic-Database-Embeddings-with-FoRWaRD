import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ARGVA
from torch_geometric.utils import train_test_split_edges
from torch_geometric.utils import from_networkx
from torch_geometric.nn import Node2Vec

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

import node2vec

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_node2vec_embedding_new(G, epochs=5, channels=100):
    embedding, model = node2vec.node2vec_embedding(G, 40, 30, 5, embed_dim=channels, neg_samples=20, batch_size=40000,
                                                   epochs=epochs)
    embedding = {n: embedding[i] for i, n in enumerate(G.nodes())}
    return embedding, model


def compute_embedding(db):
    G = db.get_row_val_graph_reg()
    embedding, model = get_node2vec_embedding_new(G, epochs=5)
    return embedding, model


if __name__ == "__main__":
    name = 'mutagenesis'
    embedding_name = 'testn3.pckl'

    path = f'Datasets/{name}'
    embedding_path = f'Embeddings/{name}/{embedding_name}'

    db = Database.load_csv(path)
    Y, rows = db.get_labels()

    scores = []
    split = StratifiedShuffleSplit(train_size=0.9, random_state=0, n_splits=10)
    for i, (train_index, test_index) in enumerate(split.split(rows, Y)):
        embedding, _ = io_utils.load_or_compute(f'{embedding_path}_{i}', lambda: compute_embedding(db))

        X_train = np.float32([embedding[rows[j]] for j in train_index])
        X_test = np.float32([embedding[rows[j]] for j in test_index])
        Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]

        clf = SVC(kernel='rbf', C=1.0)

        clf = make_pipeline(StandardScaler(), clf)

        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)

        scores.append(float(score))
        print(f"Run {i}; Accuracy: {score:.4f}")

    print(f'Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')