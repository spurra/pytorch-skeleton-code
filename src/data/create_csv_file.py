'''
    This script creates the solution .csv file from the pickle file in
    created by create_cropped_eval_data.py
'''

import pickle
import os
import numpy as np
import csv

target_dir = "./data/processed/evaluation"

anno_path = os.path.join(target_dir, 'anno_evaluation_cropped.pickle')
csv_path = os.path.join(target_dir, 'anno_evaluation_cropped.csv')

with open(anno_path, 'rb') as f:
    anno_eval = pickle.load(f)

with open(csv_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    header = ['Id']
    for i in range(0, 21):
        header += ['Joint %d x' % i]
        header += ['Joint %d y' % i]
    csv_writer.writerow(header)
    for idx,elem in enumerate(anno_eval):
        # Vectorize the joint matrix row first
        vectorized = elem.reshape(-1)
        csv_writer.writerow([idx] + vectorized.tolist())
