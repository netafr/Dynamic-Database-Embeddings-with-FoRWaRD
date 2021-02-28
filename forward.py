import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from db_utils import Database
from io_utils import save_dict
import pickle
import mmd_utils
import ek_utlis


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

eps = 1e-15
kernels = {'object': lambda x, y: x == y,
           'int64': lambda x, y: x == y,
           'float64': lambda x, y: np.exp(- ((x - y) ** 2) / (2 * ((0.05 * np.maximum(np.abs(x), np.abs(y)) + eps) ** 2)))}


kernels_dict = {}


class Forward(torch.nn.Module):

    def __init__(self, dim, num_schemes, row_idx, scheme_idx):
        super().__init__()

        self.dim = dim
        self.num_relations = num_schemes
        self.row_idx = np.vectorize(lambda x: row_idx[x])
        self.scheme_idx = scheme_idx
        self.num_tuples = len(row_idx)

        self.x = torch.nn.Parameter(torch.Tensor(self.num_tuples, dim))
        torch.nn.init.normal_(self.x, std=np.sqrt(1.0/dim))
        self.A = torch.nn.Parameter(torch.Tensor(self.num_relations, dim, dim))
        torch.nn.init.normal_(self.A)

    def forward(self, pairs, scheme_idx):
        A_sym = (self.A + torch.transpose(self.A, 2, 1)) / 2
        A = A_sym[scheme_idx]
        x_v = self.x[pairs[:, 0]].view(-1, 1, self.dim)
        x_u = self.x[pairs[:, 1]].view(-1, 1, self.dim)
        y = (x_v.matmul(A) * x_u).sum(dim=(1, 2))
        return y

    def loss(self, y_pred, y_true):
        return F.mse_loss(y_pred, y_true)

    def get_embedding(self):
        return self.x.cpu().data.numpy()

    def infer(self, old_idx, scheme_idx, y):
        old_idx = old_idx.to(device)
        scheme_idx = scheme_idx.to(device)
        y = y.to(device)

        with torch.no_grad():
            A_sym = (self.A + torch.transpose(self.A, 2, 1)) / 2
            A = A_sym[scheme_idx]
            x_old = self.x[old_idx]
            A_stack = A.matmul(x_old.reshape(-1, self.dim, 1)).view(-1, self.dim)
            A_inv = torch.pinverse(A_stack)
            x = A_inv.matmul(y)

        return x.cpu().detach().numpy()


def train(model, loader, epochs):
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for e in range(epochs):
        bar = tqdm(desc=f'Epoch {e + 1} Mean Loss: _')
        bar.reset(total=len(loader))

        epoch_losses = []
        for (pairs, vals, scheme) in loader:
            pairs, vals, scheme = pairs.to(device), vals.to(device), scheme.to(device)

            opt.zero_grad()
            y_pred = model(pairs, scheme)
            loss = model.loss(y_pred, vals)
            loss.backward()
            opt.step()

            epoch_losses.append(loss.cpu().detach().numpy())
            bar.set_description(desc=f'Epoch {e + 1} Mean Loss: {epoch_losses[-1]:.4f}')
            bar.update()

        bar.close()

    return model


def get_samples(db, depth, num_samples, sample_fct):
    tuples = [r for _, r, _ in db.iter_rows(db.predict_rel)]
    scheme_tuple_map = db.scheme_tuple_map(db.predict_rel, tuples, depth)

    samples = {}
    for scheme, tuple_map in tqdm(scheme_tuple_map.items()):
        cur_rel = scheme.split(">")[-1]
        if len(db.rel_comp_cols[cur_rel]) > 0:
            for col_id in db.rel_comp_cols[cur_rel]:
                col_kernel = kernels_dict[col_id] if col_id in kernels_dict.keys() else kernels[db.get_col_type(col_id)]
                pairs, values = sample_fct(db, col_id, tuple_map, num_samples, col_kernel)

                full_scheme = f"{scheme}>{col_id}"
                samples[full_scheme] = (pairs, values)

    return samples


def preproc_data(samples, model, batch_size):
    # stack pairs of tuples and map them to integer indices
    pairs = np.vstack([p for p, _ in samples.values()])
    pairs = torch.tensor(model.row_idx(pairs))

    # stack kernel values
    vals = torch.tensor(np.concatenate([v for _, v in samples.values()], axis=0))

    # stack schemes and map them to integer indices
    scheme = [np.int64([model.scheme_idx[s]] * samples[s][0].shape[0]) for s in samples.keys()]
    scheme = torch.tensor(np.concatenate(scheme, axis=0))

    # build torch loader for training
    data = TensorDataset(pairs, vals, scheme)
    loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=8)
    return loader


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='genes', help="Name of the data base")
    parser.add_argument("--dim", type=int, default=100, help="Dimension of the embedding")
    parser.add_argument("--depth", type=int, default=3, help="Depth of the walks")
    parser.add_argument("--kernel", type=str, default='EK', choices={'EK', 'MMD'}, help="Kernel to use for ForWaRD")
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples per start tuple and metapath")
    parser.add_argument("--batch_size", type=int, default=50000, help="Batch size during training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs during training")
    parser.add_argument("--classifier", type=str, default='SVM', choices={'NN', 'SVM'}, help="Downstream Classifier")
    parser.add_argument("--seed", type=int, default=0, help="Random Seed")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = f'Datasets/{args.data_name}'
    db = Database.load_csv(data_path)

    model_dir = f'models/{args.data_name}/{args.kernel}_{args.depth}_{args.dim}_{args.num_samples}_{args.epochs}_{args.batch_size}_{args.seed}'
    os.makedirs(model_dir, exist_ok=True)

    sample_fct = ek_utlis.ek_sample_fct if args.kernel == 'EK' else mmd_utils.mmd_sample_fct

    Y, rows = db.get_labels()

    scores = []
    split = StratifiedShuffleSplit(train_size=0.9, random_state=0, n_splits=10)
    for i, (train_index, test_index) in enumerate(split.split(rows, Y)):

        samples = get_samples(db, args.depth, args.num_samples, sample_fct)
        row_idx = {r: i for i, r in enumerate(rows)}
        scheme_idx = {s: i for i, s in enumerate(samples.keys())}
        model = Forward(args.dim, len(samples), row_idx, scheme_idx)

        loader = preproc_data(samples, model, args.batch_size)
        train(model, loader, args.epochs)

        embedding = model.get_embedding()
        embedding = {r: embedding[i] for r, i in row_idx.items()}

        X_train = np.float32([embedding[rows[j]] for j in train_index])
        X_test = np.float32([embedding[rows[j]] for j in test_index])
        Y_train, Y_test = [Y[i] for i in train_index], [Y[i] for i in test_index]

        clf = MLPClassifier(max_iter=1000) if args.classifier == 'NN' else SVC(kernel='rbf', C=1.0)
        clf = make_pipeline(StandardScaler(), clf)

        clf.fit(X_train, Y_train)
        score = clf.score(X_test, Y_test)

        scores.append(float(score))
        save_dict({'scores': scores}, f'{model_dir}/results.json')
        print(f"Run {i}; Accuracy: {score:.2f}")

    print(f'Acc: {np.mean(scores):.4f} (+-{np.std(scores):.4f})')
