import numpy as np


def mmd_sample_fct(db, col_id, tuple_map, num_samples, col_kernel, tuples_left=None, tuples_right=None):
    if tuples_left is None:
        tuples_left = list(tuple_map.keys())
    if tuples_right is None:
        tuples_right = list(tuple_map.keys())

    value_map = db.get_col_as_map(col_id, ignore_nan=True)
    value_list = {r: [value_map[s] for s in end_nodes if s in value_map.keys()] for r, end_nodes in tuple_map.items()}
    value_unique = {r: np.unique(vals, return_counts=True) for r, vals in value_list.items() if len(vals) > 0}
    value_counts = {r: {v: c for v, c in zip(vals, counts)} for r, (vals, counts) in value_unique.items()}
    pairs, values = mmd_kernel(value_counts, col_kernel, num_samples, tuples_left, tuples_right)
    return pairs, values


def mmd_kernel(value_counts, kernel, num_samples, tuples_left, tuples_right):

    filtered_left = np.array([r for r in tuples_left if r in value_counts.keys()])
    filtered_right = np.array([r for r in tuples_right if r in value_counts.keys()])
    num_left = filtered_left.shape[0]
    num_right = filtered_right.shape[0]

    if num_right * num_left == 0:
        return None, None

    if num_right <= num_samples:
        total_num_samples = num_left * (num_right-1)
        idx_left = np.arange(0, num_left).repeat(num_right-1).reshape(total_num_samples, 1)
        idx_right = (idx_left + np.tile(np.arange(1, num_right).reshape(1,-1), (num_left, 1)).reshape(-1, 1)) % num_right
    else:
        total_num_samples = num_left * num_samples
        idx_left = np.arange(0, num_left).repeat(num_samples).reshape(total_num_samples, 1)
        idx_right = (idx_left + np.random.randint(1, num_right, (total_num_samples, 1))) % num_right

    pairs = np.hstack([np.hstack([filtered_left[idx_left], filtered_right[idx_right]])])

    norm = {r: combined_kernel(vc, vc, kernel) for r, vc in value_counts.items()}
    mmd = np.float32([norm[a] + norm[b] - 2 * combined_kernel(value_counts[a], value_counts[b], kernel) for [a, b] in pairs])
    kernel_values = 1.0 - (mmd / 2.0)

    return pairs, np.float32(kernel_values)


def combined_kernel(value_counts_1, value_counts_2, kernel):
    n = sum(list(value_counts_1.values()))
    m = sum(list(value_counts_2.values()))
    res = 0.0
    for v1, c1 in value_counts_1.items():
        for v2, c2 in value_counts_2.items():
            res += c1 * c2 * kernel(v1, v2)

    return res / (n * m)