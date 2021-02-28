import glob
import os
import json
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="name of the dataset")
    args = parser.parse_args()

    for dir in glob.glob(os.path.join('models/dynamic', args.data_name, '*')):
        config = dir[:-2]
        print(config)

        ratio_paths = sorted(glob.glob(os.path.join(dir, '*')))
        for p in ratio_paths:
            ratio = float(p.split('/')[-1])
            res_file = os.path.join(p, 'results.json')
            if os.path.exists(res_file):
                with open(res_file, 'r') as f:
                    d = json.load(f)
                    mean = np.mean(d['scores'])
                    std = np.std(d['scores'])
                    print(f'{ratio}: {mean:.4f} (+-{std:.4f})')