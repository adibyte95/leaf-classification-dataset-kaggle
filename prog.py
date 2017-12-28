import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
## training data

train_data = pd.read_csv('train.csv')
X_train = train_data.drop(['id','species'], axis =1).values

#encoding the name of the plant species
label_encoder = LabelEncoder().fit(train_data['species'])
y_train =label_encoder.transform(train_data['species'])


#standerization of the features
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)

# building the logistic regression classifier that outputs probablities
params = {'C': [100, 100], 'tol':[.001,.0001]}
#solver = optimization used to reduce the error other options sgsd adam
#multi_class : indicate that the we have more than 2 classes
logistic_regression = LogisticRegression(solver = 'lbfgs', multi_class = 'multinomial')

# GridSearchCV is used to find the best value of parameters i.e one that has lowest log loss 
clf = GridSearchCV(logistic_regression, params,scoring = 'neg_log_loss',cv =5)
clf.fit(X_train, y_train)


print(clf.best_params_)
print(clf.grid_scores_)

# test the data
test_data = pd.read_csv('test.csv')
test_id = test_data.pop('id')
X_test = scaler.transform(test_data)

# predicting the probablity of each category
y_test = clf.predict_proba(X_test)

#writing the predictions to csv file
submission = pd.DataFrame(y_test, index= test_id, columns = label_encoder.classes_)
submission.to_csv('submission.csv')