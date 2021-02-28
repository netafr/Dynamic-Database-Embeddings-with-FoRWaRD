import numpy as np
import networkx as nx
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import gc

eps = 1e-15
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Node2Vec(nn.Module):

    def __init__(self, node_size, embed_dim, dyn_filter=None):
        super().__init__()
        self.node_size = node_size
        self.embed_dim = embed_dim

        self.start_embeds = torch.nn.Parameter(torch.Tensor(node_size, embed_dim))
        torch.nn.init.normal_(self.start_embeds, std=np.sqrt(1.0 / embed_dim))

        self.dyn_filter = dyn_filter

    def forward(self, start_node, pos_samples, neg_samples):
        neg_samples = torch.randint(0, self.node_size, neg_samples.shape)

        start_emb = self.start_embeds[start_node]
        pos_emb = self.start_embeds[pos_samples]
        neg_emb = self.start_embeds[neg_samples]

        pos_dot = (start_emb * pos_emb).sum(dim=-1)
        neg_dot = (start_emb * neg_emb).sum(dim=-1)

        pos_prob = torch.sigmoid(pos_dot)
        neg_prob = torch.sigmoid(neg_dot)
        loss = - torch.log(pos_prob + eps).mean(dim=1) - torch.log(1.0 - neg_prob + eps).mean(dim=1)

        return loss

    def get_embedding(self):
        return self.start_embeds.data.cpu().numpy()


class MP2Vec(nn.Module):

    def __init__(self, node_size, embed_dim, node_types):
        super().__init__()
        self.node_size = node_size
        self.embed_dim = embed_dim
        self.num_types = np.max(node_types) + 1
        self.node_types = torch.tensor(node_types)

        self.start_embeds = torch.nn.Parameter(torch.Tensor(node_size, embed_dim))
        torch.nn.init.normal_(self.start_embeds)
        self.end_embeds = torch.nn.Parameter(torch.Tensor(node_size, self.num_types, embed_dim))
        torch.nn.init.normal_(self.end_embeds)

    def forward(self, start_node, pos_samples, neg_samples):
        start_emb = self.start_embeds[start_node]
        node_type = self.node_types[start_node]

        pos_emb = self.end_embeds[pos_samples, node_type]
        neg_emb = self.end_embeds[neg_samples, node_type]

        pos_dot = (start_emb * pos_emb).sum(dim=-1)
        neg_dot = (start_emb * neg_emb).sum(dim=-1)

        pos_prob = torch.sigmoid(pos_dot)
        neg_prob = torch.sigmoid(neg_dot)
        loss = - torch.log(pos_prob + eps).mean(dim=1) - torch.log(1.0 - neg_prob + eps).mean(dim=1)

        return loss

    def get_embedding(self):
        return self.start_embeds.data.numpy()


def train(model, start, pos_samples, neg_samples, batch_size, epochs):
    batch_count = int(np.ceil(start.shape[0] / batch_size))

    shuffle = np.random.permutation(start.shape[0])
    start, pos_samples, neg_samples = (x[shuffle] for x in (start, pos_samples, neg_samples))
    start, pos_samples, neg_samples = (torch.tensor(x).to(device) for x in (start, pos_samples, neg_samples))
    start, pos_samples, neg_samples = (torch.split(x, batch_size) for x in (start, pos_samples, neg_samples))

    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for e in range(epochs):
        bar = tqdm(desc=f'Epoch {e + 1} Mean Loss: _')
        bar.reset(total=batch_count)

        epoch_losses = []
        for start_batch, pos_batch, neg_batch in zip(start, pos_samples, neg_samples):
            opt.zero_grad()
            wanted_part = model.start_embeds[model.dyn_filter].clone().detach()
            loss = model(start_batch, pos_batch, neg_batch)
            loss = loss.mean()
            # loss = - torch.log(pos_prob).mean()
            loss.backward()
            opt.step()

            if model.dyn_filter is not None:
                with torch.no_grad():
                    model.start_embeds[model.dyn_filter] = wanted_part.clone().requires_grad_(False)

            epoch_losses.append(loss.detach().cpu().numpy())
            bar.set_description(desc=f'Epoch {e + 1} Mean Loss: {np.mean(epoch_losses):.4f}')
            bar.update()

        bar.close()


def dynamic_train(model, start, pos_samples, neg_samples, batch_size, epochs):
    batch_count = int(np.ceil(start.shape[0] / batch_size))

    shuffle = np.random.permutation(start.shape[0])
    start, pos_samples, neg_samples = (x[shuffle] for x in (start, pos_samples, neg_samples))
    start, pos_samples, neg_samples = (torch.tensor(x).to(device) for x in (start, pos_samples, neg_samples))
    start, pos_samples, neg_samples = (torch.split(x, batch_size) for x in (start, pos_samples, neg_samples))

    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    for e in range(epochs):
        bar = tqdm(desc=f'Epoch {e + 1} Mean Loss: _')
        bar.reset(total=batch_count)

        epoch_losses = []
        for start_batch, pos_batch, neg_batch in zip(start, pos_samples, neg_samples):
            opt.zero_grad()
            wanted_part = model.start_embeds[~model.dyn_filter].clone().detach()
            loss = model(start_batch, pos_batch, neg_batch)
            loss = loss.mean()
            loss.backward()
            opt.step()

            with torch.no_grad():
                model.start_embeds[~model.dyn_filter] = wanted_part.clone().requires_grad_(False)

            epoch_losses.append(loss.detach().cpu().numpy())
            bar.set_description(desc=f'Epoch {e + 1} Mean Loss: {np.mean(epoch_losses):.4f}')
            bar.update()

        bar.close()


def compute_walks(g, walks_per_node, steps, p, q):
    nodes = np.arange(0, g.order())
    # we don't use graph.nodes() as we don't need the actual names, but only their indices for the adjacency list
    q_frac = 1 / q

    # in node2vec, random walk probabilities are guided according to p,q
    # 1/p for the last node
    # 1   for 1-hop neighbors of the old node
    # 1/q for 2-hop neighbors of the old node

    # for computational reasons we use the factors 1,p,p/q which preserves the relative importance of each node

    adjacency_matrix = nx.linalg.adjacency_matrix(g).todense()

    walks = []

    # once for every random walk select a starting vertex
    for n in tqdm(nodes):

        for _ in range(walks_per_node):

            current_node = n
            last_node = n
            walk = [n]

            # perform a random walk
            for _ in range(steps):
                # get neighbors of old and current node
                current_neighbors = adjacency_matrix[current_node]
                old_neighbors = adjacency_matrix[last_node]

                # compute transition probabilities
                current_neighbors = q_frac * p * current_neighbors
                old_neighbors = q * old_neighbors
                neighbors = np.array(np.multiply(current_neighbors, old_neighbors)).flatten()
                neighbors[last_node] = 1

                # perform the random step
                last_node = current_node
                probabilities = np.reshape(neighbors / neighbors.sum(), (-1,))
                current_node = np.random.choice(nodes, p=probabilities)

                walk.append(current_node)

            walks.append(walk)

    walks = np.int64(walks)
    start = walks[:, 0].reshape(-1, 1)
    pos_samples = walks[:, 1:]
    return start, pos_samples


def compute_walks_sparse(g, walks_per_node, steps, context_size):
    nodes = np.arange(0, g.order())

    adj = nx.linalg.adjacency_matrix(g).tolil()
    degrees = np.int64([d for _, d in nx.degree(g)])

    walks = []

    for n in tqdm(nodes):
        random_choices = np.random.randint(0, np.iinfo(np.int64).max, (walks_per_node, steps))

        current_node = np.int64([n for _ in range(walks_per_node)])
        walk = [current_node]

        # perform a random walk
        for step in range(steps):
            neighbors = adj.rows[current_node]
            choices = random_choices[:, step] % degrees[current_node]
            current_node = [n[c] for n, c in zip(neighbors, choices)]
            walk.append(current_node)

        walks.append(np.int64(walk).transpose())

    walks = np.vstack(walks)
    start, pos_samples = split_walks(walks, context_size)
    return start, pos_samples


def split_walks(walks, context_size):
    short_walks = []
    full_size = context_size + 1

    chunks = walks.shape[1] - full_size  # // full_size
    for c in range(chunks):
        i = c  # * full_size
        short_walks.append(walks[:, i:i + full_size])

    short_walks = np.vstack(short_walks)
    start = short_walks[:, 0].reshape(-1, 1)
    pos_samples = short_walks[:, 1:]
    return start, pos_samples


def get_negative_samples(g, start, pos_samples, neg_sample_count):
    # walks = np.hstack([start, pos_samples])
    # num_walks = start.shape[0]
    # nodes = g.order()
    # negative_samples = []
    # for w in range(num_walks):
    #    walk = walks[w]
    #    p = np.ones((1, nodes), dtype=np.float32).flatten()
    #    p[walk] = 0.0
    #    p /= np.sum(p)
    #    neg = np.random.choice(nodes, size=neg_sample_count, replace=False, p=p)
    #    negative_samples.append(neg)

    # return np.int64(negative_samples)
    return np.random.randint(0, g.order(), (start.shape[0], neg_sample_count))


def node2vec_embedding(g, walks_per_node, steps, context_size, embed_dim=300, neg_samples=5, batch_size=128, epochs=10):
    start, pos_samples = compute_walks_sparse(g, walks_per_node, steps=steps, context_size=context_size)
    neg_samples = get_negative_samples(g, start, pos_samples, neg_samples)

    model = Node2Vec(g.order(), embed_dim)
    train(model, start, pos_samples, neg_samples, batch_size=batch_size, epochs=epochs)
    embedding = model.get_embedding()
    return embedding, model

def get_tensors(only_cuda=False, omit_objs=[]):
    """
    :return: list of active PyTorch tensors
    >>> import torch
    >>> from torch import tensor
    >>> clean_gc_return = map((lambda obj: del_object(obj)), gc.get_objects())
    >>> device = "cuda" if torch.cuda.is_available() else "cpu"
    >>> device = torch.device(device)
    >>> only_cuda = True if torch.cuda.is_available() else False
    >>> t1 = tensor([1], device=device)
    >>> a3 = tensor([[1, 2], [3, 4]], device=device)
    >>> # print(get_all_tensor_names())
    >>> tensors = [tensor_obj for tensor_obj in get_tensors(only_cuda=only_cuda)]
    >>> # print(tensors)
    >>> # We doubled each t1, a3 tensors because of the tensors collection.
    >>> expected_tensor_length = 2
    >>> assert len(tensors) == expected_tensor_length, f"Expected length of tensors {expected_tensor_length}, but got {len(tensors)}, the tensors: {tensors}"
    >>> exp_size = (2,2)
    >>> act_size = tensors[1].size()
    >>> assert exp_size == act_size, f"Expected size {exp_size} but got: {act_size}"
    >>> del t1
    >>> del a3
    >>> clean_gc_return = map((lambda obj: del_object(obj)), tensors)
    """
    add_all_tensors = False if only_cuda is True else True
    # To avoid counting the same tensor twice, create a dictionary of tensors,
    # each one identified by its id (the in memory address).
    tensors = {}

    # omit_obj_ids = [id(obj) for obj in omit_objs]

    def add_tensor(obj):
        if torch.is_tensor(obj):
            tensor = obj
        elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
            tensor = obj.data
        else:
            return

        if (only_cuda and tensor.is_cuda) or add_all_tensors:
            tensors[id(tensor)] = tensor

    for obj in gc.get_objects():
        try:
            # Add the obj if it is a tensor.
            add_tensor(obj)
            # Some tensors are "saved & hidden" for the backward pass.
            if hasattr(obj, 'saved_tensors') and (id(obj) not in omit_objs):
                for tensor_obj in obj.saved_tensors:
                    add_tensor(tensor_obj)
        except Exception as ex:
            pass
            # print("Exception: ", ex)
            # logger.debug(f"Exception: {str(ex)}")
    return tensors.values()  # return a list of detected tensors

def node2vec_dynamic_embedding(g, static_g, walks_per_node, steps, context_size, embed_dim=300, neg_samples=5,
                               batch_size=128, epochs_static=5, epochs_dynamic=5):
    G_diff = g.copy()
    G_diff.remove_nodes_from(n for n in g if n in static_g)

    static_nodes = list(static_g)
    dynamic_nodes = list(G_diff)
    dyn_filter = np.zeros((g.order()), dtype=bool)
    static_to_full = {}
    dynamic_to_full = {}
    for index, node in enumerate(g.nodes()):
        if node in G_diff:
            dyn_filter[index] = True
            dynamic_to_full[dynamic_nodes.index(node)] = index
        if node in static_g:
            static_to_full[static_nodes.index(node)] = index

    static_to_full_vectorized = np.vectorize(lambda x: static_to_full[x])

    model = Node2Vec(g.order(), embed_dim, dyn_filter)

    start, pos_samples = compute_walks_sparse(static_g, walks_per_node, steps=steps, context_size=context_size)
    neg_samples_s = get_negative_samples(static_g, start, pos_samples, neg_samples)
    start = np.array([[static_to_full[x[0]]] for x in start])

    pos_samples = static_to_full_vectorized(pos_samples)
    neg_samples_s = static_to_full_vectorized(neg_samples_s)
    train(model, start, pos_samples, neg_samples_s, batch_size=batch_size, epochs=epochs_static)
    tensors = get_tensors(True)
    for ten in tensors:
        del ten
    gc.collect()
    torch.cuda.empty_cache()
    start, pos_samples = compute_walks_sparse(g, walks_per_node, steps=steps, context_size=context_size)
    neg_samples_s = get_negative_samples(g, start, pos_samples, neg_samples)
    dynamic_train(model, start, pos_samples, neg_samples_s, batch_size=batch_size, epochs=epochs_dynamic)
    embedding = model.get_embedding()
    return embedding, model
