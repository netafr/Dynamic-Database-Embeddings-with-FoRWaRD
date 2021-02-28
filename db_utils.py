import pandas as pd
import numpy as np
import networkx as nx
import glob
import os
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from itertools import chain


class Database:

    def __init__(self, relations, column_map, predict_col):

        self.relations = relations

        self.column_map = column_map
        self.col_ids = list(column_map.keys())
        self.base_col_ids = list(column_map.values())
        self.num_base_columns = len(self.base_col_ids)

        self.predict_col = column_map[predict_col]
        self.predict_rel = predict_col.split('@')[1]

        self.init_maps()

    def init_maps(self):
        self.foreign_cols = []
        self.comp_cols = []
        self.rel_comp_cols = {r: [] for r, _ in self.iter_rel()}
        self.arrows_out = {r: [] for r, _ in self.iter_rel()}
        self.arrow_rel_map = {}
        self.arrow_row_map = {}
        self.inv_arrow = {}

        for k, v in self.column_map.items():
            if k != v:
                self.foreign_cols += [k, v]
                rel_k = k.split('@')[1]
                rel_v = v.split('@')[1]
                arrow_kv = f'{k}-{v}'
                arrow_vk = f'{v}-{k}'
                self.inv_arrow[arrow_kv] = arrow_vk
                self.inv_arrow[arrow_vk] = arrow_kv
                self.arrow_rel_map[arrow_kv] = rel_v
                self.arrow_rel_map[arrow_vk] = rel_k
                self.arrows_out[rel_k].append(arrow_kv)
                self.arrows_out[rel_v].append(arrow_vk)

                k_val_dict = self.get_col_as_map(k)
                v_val_dict = self.get_col_as_map(v)
                v_val_inv = {val: row for row, val in v_val_dict.items()}

                self.arrow_row_map[arrow_kv] = {row: [v_val_inv[value]] if value in v_val_inv.keys() else [] for
                                                row, value in k_val_dict.items()}
                self.arrow_row_map[arrow_vk] = {row: [] for row in v_val_dict.keys()}
                for row, val in k_val_dict.items():
                    if val in v_val_inv.keys():
                        self.arrow_row_map[arrow_vk][v_val_inv[val]].append(row)

        for col in [col for col in self.col_ids if col not in self.foreign_cols and col != self.predict_col]:
            self.comp_cols.append(col)
            self.rel_comp_cols[col.split('@')[1]].append(col)

    def get_num_rows(self):
        return len([r for _, r, _ in self.iter_rows()])

    def get_col_as_map(self, column, ignore_nan=True):
        col, rel = column.split('@')
        relation = self.relations[rel]
        column_frame = relation[col]
        if ignore_nan:
            row_val_map = {f'{i}@{rel}': v for i, v in enumerate(column_frame) if not pd.isnull(v)}
        else:
            row_val_map = {f'{i}@{rel}': v for i, v in enumerate(column_frame)}
        return row_val_map

    def get_arrow_as_map(self, column, origin):
        col_map = self.get_col_as_map(column)
        org_map = self.get_col_as_map(origin)
        org_map_inv = {v: r for r, v in org_map.items()}
        arrow_map = {r: org_map_inv[v] for r, v in col_map.items() if v in org_map_inv.keys()}
        return arrow_map

    def get_col_type(self, column):
        col, rel = column.split('@')
        relation = self.relations[rel]
        dtype = relation.dtypes[col]
        return dtype.name

    def get_row_classes(self):
        num_classes = 0
        row_classes = {}
        class_idx = {}

        pred_col, pred_rel = self.predict_col.split('@')
        df = self.relations[pred_rel]
        for id, cell in enumerate(df[pred_col]):
            row_id = f'{id}@{pred_rel}'

            if cell not in class_idx.keys():
                class_idx[cell] = num_classes
                num_classes += 1

            row_classes[row_id] = class_idx[cell]
        return row_classes


    @staticmethod
    def get_col_id(rel_name, col_name):
        return f'{col_name}@{rel_name}'

    @staticmethod
    def read_col_spec(path, relations):
        col_map = {}
        predict = None

        all_col_ids = []
        for rel_name, rel in relations.items():
            all_col_ids += [Database.get_col_id(rel_name, col_name) for col_name, _ in rel.iteritems()]

        for id in all_col_ids:
            if id not in col_map.keys():
                col_map[id] = id

        with open(path, 'r') as f:
            for line in f:
                values = line.split()

                if len(values) == 4:
                    col_from = Database.get_col_id(values[0], values[1])
                    col_to = Database.get_col_id(values[2], values[3])
                    col_map[col_from] = col_to

                elif len(values) == 3 and values[0] == 'predict':
                    predict = Database.get_col_id(values[1], values[2])

        return col_map, predict

    @staticmethod
    def load_csv(path, index_col=0):
        relation_paths = glob.glob(os.path.join(path, '*.csv'))
        col_spec_path = glob.glob(os.path.join(path, '*cols'))[0]

        #relations = {os.path.basename(p).split('.')[0]: pd.read_csv(p, index_col=index_col) for p in relation_paths}

        relations = {os.path.basename(p).split('.')[0]: pd.read_csv(p,
                                                                    na_values=['N/A', '#N/A N/A', '\#NA', '-1.#IND',
                                                                               '-1.#QNAN', '-NaN', '-nan', '<NA>',
                                                                               'N/A', 'NULL', 'NaN', 'n/a', 'nan',
                                                                               'null', '?'],
                                                                    keep_default_na=False)
                     for p in relation_paths}

        col_map, predict = Database.read_col_spec(col_spec_path, relations)

        db = Database(relations, col_map, predict)
        return db
        
    def bin_float_columns(self, bins=5):
        for rel_id, df in self.iter_rel():
            for column in df:
                if df[column].dtype == 'float64':
                    values = np.nan_to_num(np.reshape(np.float64(df[column]), (-1, 1)))
                    
                    num_unique = len(np.unique(values))
                    new_bins = max(bins, num_unique//10)
                    
                    #new_bins = bins
                    kmeans = KMeans(n_clusters=bins, random_state=0).fit(values)
                    clusters = kmeans.predict(values)
                    df[column] = clusters

    def get_labels(self):
        col, rel = self.predict_col.split('@')
        df = self.relations[rel]
        Y = df[col]
        rows = [f'{id}@{rel}' for id, _ in df.iterrows()]
        return Y, rows

    def iter_rel(self):
        for rel_id, df in self.relations.items():
            yield rel_id, df

    def iter_rows(self, rel_id=None, partition=None):
        if rel_id is None:
            for rel_id, df in self.iter_rel():
                for id, row in df.iterrows():
                    row_id = f'{id}@{rel_id}'
                    if partition is None or self.tuple_partition[row_id] <= partition:
                        yield rel_id, row_id, row
        else:
            for id, row in self.relations[rel_id].iterrows():
                row_id = f'{id}@{rel_id}'
                if partition is None or self.tuple_partition[row_id] <= partition:
                    yield rel_id, row_id, row

    def iter_cells(self, rel_id=None):
        for rel_id, row_id, row in self.iter_rows(rel_id):
            for col, cell in row.items():
                col_id = f'{col}@{rel_id}'
                col_id = self.column_map[col_id]
                yield rel_id, row_id, col_id, str(cell)

    def get_sigmod_graph(self, feature_size=300, add_row_classes=False, only_pred_relation=False, add_relation_nodes=False):
        # model = FastText.load_fasttext_format("cc.en.300.bin")

        cell_counts = {col_id: {} for col_id in self.base_col_ids}
        for rel_id, row_id, col_id, cell in self.iter_cells():
            if cell in cell_counts[col_id].keys():
                cell_counts[col_id][cell] += 1
            else:
                cell_counts[col_id][cell] = 1

        G = nx.Graph()

        col_nodes = [col_id for col_id in self.base_col_ids if not col_id == self.predict_col]
        col_features = {c: np.float32(np.random.normal(0.0, 1.0, feature_size)) for c in col_nodes}
        G.add_nodes_from([(c, {'x': col_features[c], 'type': 'col'}) for c in col_nodes])

        row_nodes = [row_id for _, row_id, _ in self.iter_rows(self.predict_rel if only_pred_relation else None)]
        row_features = {r: np.float32(np.random.normal(0.0, 1.0, feature_size)) for r in row_nodes}
        G.add_nodes_from([(r, {'x': row_features[r], 'type': 'row'}) for r in row_nodes])

        if add_relation_nodes:
            rel_nodes = [rel_id for rel_id in self.relations.keys()]
            rel_features = {r: np.float32(np.random.normal(0.0, 1.0, feature_size)) for r in rel_nodes}
            G.add_nodes_from([(c, {'x': rel_features[c], 'type': 'rel'}) for c in rel_nodes])

        if add_row_classes:
            row_classes = self.get_row_classes()
            nx.set_node_attributes(G, row_classes, 'y')

        for rel_id, row_id, col_id, cell in self.iter_cells(self.predict_rel if only_pred_relation else None):
            if not col_id == self.predict_col:
                values = cell.split() if cell_counts[col_id][cell] == 1 else [cell]
                for val in values:
                    val_id = f'{val}@{col_id}'
                    if not G.has_node(val_id):
                        node_features = np.float32(np.random.normal(0.0, 1.0, feature_size))
                        G.add_node(val_id, x=node_features, type='val')
                        G.add_edge(val_id, col_id)

                    G.add_edge(row_id, val_id)
                    if add_relation_nodes:
                        G.add_edge(rel_id, val_id)
                        G.add_edge(rel_id, row_id)

        return G

    def get_row_val_graph(self, add_row_classes=True, partition=1):

        cell_counts = {col_id: {} for col_id in self.base_col_ids}
        for rel_id, row_id, col_id, cell in self.iter_cells():
            if cell in cell_counts[col_id].keys():
                cell_counts[col_id][cell] += 1
            else:
                cell_counts[col_id][cell] = 1

        G = nx.Graph()

        row_nodes = [row_id for _, row_id, _ in self.iter_rows() if self.tuple_partition[row_id] <= partition]
        G.add_nodes_from([(r, {'type': 'row'}) for r in row_nodes])

        for rel_id, row_id, col_id, cell in self.iter_cells():
            if (not col_id == self.predict_col) and self.tuple_partition[row_id] <= partition:
                values = cell.split() if cell_counts[col_id][cell] == 1 else [cell]
                for val in values:
                    val_id = f'{val}@{col_id}'
                    if not G.has_node(val_id):
                        G.add_node(val_id, type='val')

                    G.add_edge(row_id, val_id)

        return G

    def scheme_tuple_map(self, rel_id, start_rows, depth, arrow_in=None, partition=None):
        if partition is not None:
            start_rows = [s for s in start_rows if self.tuple_partition[s] <= partition]

        tuple_map = {f'{rel_id}': {r: {r} for r in start_rows}}

        if depth > 0:

            if arrow_in is not None:
                arrows_out = [a for a in self.arrows_out[rel_id] if not a == self.inv_arrow[arrow_in]]
            else:
                arrows_out = self.arrows_out[rel_id]

            for arrow_out in arrows_out:
                rel_id_out = self.arrow_rel_map[arrow_out]
                rows = set(chain.from_iterable(self.arrow_row_map[arrow_out][r] for r in start_rows))

                if partition is not None:
                    rows = [r for r in rows if self.tuple_partition[r] <= partition]

                arrow_mpr_map = self.scheme_tuple_map(rel_id_out, rows, depth - 1, arrow_out)

                for meta_path, row_map in arrow_mpr_map.items():
                    tuple_map[f'{arrow_out}>{meta_path}'] = {r: list(chain.from_iterable(row_map[s] for s in self.arrow_row_map[arrow_out][r])) for r in start_rows}

        return tuple_map

    def iter_rel_bfs(self):
        buffer = [self.predict_rel]
        depth = [0]
        i = 0
        while i<len(buffer):
            cur_rel = buffer[i]
            cur_neighbours = [a.split('@')[-1] for a in self.arrows_out[cur_rel]]
            new_rel = [n for n in cur_neighbours if n not in buffer]
            buffer += new_rel
            depth += [depth[i] + 1] * len(new_rel)
            i += 1

        return iter(zip(buffer[1:], depth[1:]))

    def partition(self, partition_prob=None, partition=None):
        self.tuple_partition = {}
        partitioned_relations = []

        if partition_prob is not None:
            self.num_partitions = len(partition_prob)
            pred_tuples = [t for _, t, _ in self.iter_rows(self.predict_rel)]

            num_pred_tuples = len(pred_tuples)
            partition_count = [int(num_pred_tuples*p) for p in partition_prob]

            np.random.shuffle(pred_tuples)
            low = 0
            high = partition_count[0] + 1
            for i in range(self.num_partitions):
                for t in pred_tuples[low:high]:
                    self.tuple_partition[t] = i

                low = high
                high = high + partition_count[i+1] if i < self.num_partitions-1 else num_pred_tuples

        else:
            for t, p in partition.items():
                self.tuple_partition[t] = p

        partitioned_relations.append(self.predict_rel)

        cur_depth = 1
        rel_buffer = []
        for rel_id, depth in self.iter_rel_bfs():

            if cur_depth < depth:
                cur_depth += 1
                partitioned_relations += rel_buffer
                rel_buffer = []

            back_arrows = [a for a in self.arrows_out[rel_id] if a if a.split('@')[-1] in partitioned_relations]

            for _, t, _ in self.iter_rows(rel_id):
                parent_tuples = list(chain.from_iterable(self.arrow_row_map[arrow][t] for arrow in back_arrows))
                parent_partitions = [self.tuple_partition[s] for s in parent_tuples if s in self.tuple_partition.keys()]
                self.tuple_partition[t] = np.min(parent_partitions) if len(parent_partitions) > 0 else 0

            rel_buffer.append(rel_id)


