import numpy as np
MAXINT = np.iinfo(np.int64).max


def ek_sample_fct(db, col_id, tuple_map, num_samples, col_kernel, tuples_left=None, tuples_right=None):
    if tuples_left is None:
        tuples_left = list(tuple_map.keys())
    if tuples_right is None:
        tuples_right = list(tuple_map.keys())

    # get values and filter out rows that have no walk in the scheme
    value_map = db.get_col_as_map(col_id, ignore_nan=True)
    value_map = {r: [value_map[s] for s in end_nodes if s in value_map.keys()] for r, end_nodes in tuple_map.items()}
    filtered_left = np.array([r for r in tuples_left if len(value_map[r]) > 0])
    filtered_right = np.array([r for r in tuples_right if len(value_map[r]) > 0])
    num_left = filtered_left.shape[0]
    num_right = filtered_right.shape[0]
    total_num_samples = num_left * num_samples

    if num_right <= 1 or num_left <= 1:
        return None, None

    # sampled pairs of distinct tuples
    idx_left = np.arange(0, num_left).repeat(num_samples).reshape(total_num_samples, 1)
    idx_right = (idx_left + np.random.randint(1, num_right, (total_num_samples, 1))) % num_right
    pairs = np.hstack([np.hstack([filtered_left[idx_left], filtered_right[idx_right]])])

    # choose random destination value for each tuple and compute kernel
    choose_val = np.vectorize(lambda r, c: value_map[r][c % len(value_map[r])])
    compute_kernel = np.vectorize(lambda a, b: col_kernel(a, b))
    val_choices = np.random.randint(0, MAXINT, (pairs.shape[0], 2))
    val_pairs = choose_val(pairs, val_choices)
    y = np.float32(compute_kernel(val_pairs[:,0], val_pairs[:,1]))
    return pairs, y
