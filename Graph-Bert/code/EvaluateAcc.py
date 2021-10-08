'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import csv
import numpy as np
from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class EvaluateAcc(evaluate):
    data = None
    
    def evaluate(self):
        acc_scr = accuracy_score(self.data['true_y'], self.data['pred_y'])
        f1_scr = f1_score(self.data['true_y'], self.data['pred_y'], average = 'weighted', labels=np.unique(self.data['pred_y']))	
        # precision = precision_score(self.data['true_y'], self.data['pred_y'])
        # recall = recall_score(self.data['true_y'], self	.data['pred_y'])
        return(acc_scr, f1_scr)

    # def test(self):
    # 	lines = []
    # 	for i in range(len(self.data['true_y'])):
    # 		lines.append([str(self.data['true_y'][i].numpy()), str(self.data['pred_y'][i].numpy())])
    # 	with open("visualise.csv", "w") as f:
    # 		writer = csv.writer(f)
    # 		writer.writerow(["True_y", "Pred_y"])
    # 		for row in lines:
    # 			writer.writerow(row)