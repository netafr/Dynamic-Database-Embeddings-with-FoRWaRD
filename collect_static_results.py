import glob
import os
import json
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, help="name of the dataset")
    args = parser.parse_args()

    for dir in glob.glob(os.path.join('models', args.data_name, '*')):
        res_file = os.path.join(dir, 'results.json')

        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                d = json.load(f)
                if 'scores' in d.keys():
                    print(f'{dir}: {np.mean(d["scores"]):.4f} {np.std(d["scores"]):.4f}')