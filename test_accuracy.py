# Andy Liu

import pandas as pd
import sklearn.metrics
import sys

def main(pred_results_path):
	correct_results_path = "handouts/Correctresults_CV.csv"

	true_labels = pd.read_csv(correct_results_path)
	predictions = pd.read_csv(pred_results_path)
	test_acc = sklearn.metrics.accuracy_score(true_labels, predictions)

	print(test_acc)

if __name__ == '__main__':
	pred_results_path = sys.argv[1]
	main(pred_results_path)