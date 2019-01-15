from scripts.data import MNISTData
from scripts.model import DigitClassifier


#  --------------
# Step 1
# getting data and familiarity with it

data_obj = MNISTData()
X_train, y_train = data_obj.get_train_set()


#  --------------
# Step 2
# training classifier with default data
clf_o = DigitClassifier()
clf = clf_o.train_rnd_forest_clf(X_train, y_train, 'RandomForestClassifier (default data)')


#  --------------
# Step 3
# getting test data sets and model evaluation
X_test, y_test = data_obj.get_test_sets()
clf_o.evaluate_model(clf, X_test, y_test)


#  --------------
# Step 4
# reducing dimensionality for X_train
X_train_reduced, X_test_reduced = data_obj.get_reduced_dimensionality_data(X_train, X_test)


#  --------------
# Step 5
# training classifier with reduced data and model evaluation
clf_2 = clf_o.train_rnd_forest_clf(X_train_reduced, y_train, 'RandomForestClassifier (reduced data)')
clf_o.evaluate_model(clf_2, X_test_reduced, y_test)







