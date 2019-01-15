from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import time


class DigitClassifier:

	def train_rnd_forest_clf(self, X_train, y_train, txt_header):
		start_time = time.time()
		print(txt_header)
		print('Train was started...')
		clf = RandomForestClassifier() # n_jobs=-1 removed to see how to change time costs
		clf.fit(X_train, y_train)
		y_train_pred = cross_val_predict(clf, X_train, y_train, cv=3)
		self._calculate_model_metrics(y_train, y_train_pred)
		end_time = time.time()
		print('Total train time = ', round(end_time - start_time), 's')
		return clf

	@staticmethod
	def evaluate_model(model, X_test, y_test):
		y_pred = model.predict(X_test)
		print('Model accuracy = ', accuracy_score(y_test, y_pred))
		print('------------------------------------------------------')

	@staticmethod
	def _calculate_model_metrics(y_train, y_pred):
		print('Calculating metrics...')
		labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
		precision, recall, fscore, support = precision_recall_fscore_support(
			y_train, y_pred,
			labels=labels)

		precision = np.reshape(precision, (10, 1))
		recall = np.reshape(recall, (10, 1))
		fscore = np.reshape(fscore, (10, 1))
		data = np.concatenate((precision, recall, fscore), axis=1)
		df = pd.DataFrame(data)
		df.columns = ['Precision', 'Recall', 'Fscore']
		print(df)

		print('\n Average values')
		print('Precision = ', df['Precision'].mean())
		print('Recall = ', df['Recall'].mean())
		print('F1 score = ', df['Fscore'].mean())
