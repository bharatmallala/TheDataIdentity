
# Project

import pandas as pd
import numpy as np

#reading data file
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

#ordianal values mapping

train_data['difficulty_level'] = train_data['difficulty_level'].map({'easy': 1,'intermediate':2, 'hard':3, 'vary hard' :4})
train_data['education'] = train_data['education'].map({'No Qualification': 1,'High School Diploma':2, 'Matriculation':3, 'Bachelors' :4, 'Masters':5})

test_data['difficulty_level'] = test_data['difficulty_level'].map({'easy': 1,'intermediate':2, 'hard':3, 'vary hard' :4})
test_data['education'] = test_data['education'].map({'No Qualification': 1,'High School Diploma':2, 'Matriculation':3, 'Bachelors' :4, 'Masters':5})


#diffentiating Y and X
X = train_data.iloc[:, :-1].values
Y = train_data.iloc[:, 15].values

X1 = test_data.iloc[:,:].values
#imputing missing values

from sklearn.preprocessing import Imputer
train_data.isnull().sum()
imputer1 = Imputer(missing_values = 'NaN', strategy  = 'mean', axis = 0)
imputer2 = Imputer(missing_values = 'NaN', strategy  = 'most_frequent', axis = 0)  

imputer1.fit(X[:, 11:12])
imputer2.fit(X[:,14:15])
X[:,11:12] = imputer1.transform(X[:, 11:12])
X[:,14:15] = imputer2.transform(X[:, 14:15])

test_data.isnull().sum()
imputer1.fit(X1[:, 11:12])
imputer2.fit(X1[:,14:15])
X1[:,11:12] = imputer1.transform(X1[:, 11:12])
X1[:,14:15] = imputer2.transform(X1[:, 14:15])

count = 0
for point in X[:,11]:
    if np.isnan(point):
        count = count+1
print(count)

#dealing with  categorical varaibles
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

for i in range(0,15):
    print(len(set(X1[:,i])))
    
X = X[:,[1,2,3,4,5,6,8,9,10,11,12,13,14]]
X_id = X1[:,0]
X1 = X1[:,[1,2,3,4,5,6,8,9,10,11,12,13,14]]

#label encoding
label = LabelEncoder()

X[:,0] = label.fit_transform(X[:,0])
X[:,1] = label.fit_transform(X[:,1])
X[:,3] = label.fit_transform(X[:,3])
X[:,4] = label.fit_transform(X[:,4])
X[:,5] = label.fit_transform(X[:,5])
X[:,6] = label.fit_transform(X[:,6])
X[:,7] = label.fit_transform(X[:,7])
X[:,11] = label.fit_transform(X[:,11])

X1[:,0] = label.fit_transform(X1[:,0])
X1[:,1] = label.fit_transform(X1[:,1])
X1[:,3] = label.fit_transform(X1[:,3])
X1[:,4] = label.fit_transform(X1[:,4])
X1[:,5] = label.fit_transform(X1[:,5])
X1[:,6] = label.fit_transform(X1[:,6])
X1[:,7] = label.fit_transform(X1[:,7])
X1[:,11] = label.fit_transform(X1[:,11])

#onehot encoding
onehot = OneHotEncoder(categorical_features=[0])
X = onehot.fit_transform(X).toarray()
X1= onehot.fit_transform(X1).toarray()
X = np.delete(X,0, axis = 1)
X1 = np.delete(X1,0, axis = 1)

onehot1 = OneHotEncoder(categorical_features= [21])
X = onehot1.fit_transform(X).toarray()
X = np.delete(X,21, axis = 1)
X1 = onehot1.fit_transform(X1).toarray()
X1 = np.delete(X1,21, axis = 1)


#train test split
from sklearn.model_selection import train_test_split
X_train,X_test, Y_train, Y_test =  train_test_split(X,Y, test_size = 0.2 , random_state  = 0)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#Logistic Regression
from sklearn.linear_model import LogisticRegression
lv = LogisticRegression(random_state=0)
lv.fit(X_train, Y_train)
lv_pred = lv.predict(X_test)

#knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_test)

#SVC
from sklearn.svm import SVC
sv = SVC(kernel='linear', random_state= 0)
sv.fit(X_train, Y_train)
sv_pred = sv.predict(X_test)

#kernel svm
svc = SVC(kernel= 'rbf', random_state= 0)
svc.fit(X_train, Y_train)
svc_pred = svc.predict(X_test)

#decison tress
from sklearn.tree import DecisionTreeClassifier
dc = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dc.fit(X_train, Y_train)
dc_pred = dc.predict(X_test)

#random forest

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators= 100, criterion= 'entropy', random_state=0)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
rf.fit(X,Y)
rf_pred1 = rf.predict(X1)
#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

ann = Sequential()
ann.add(Dense(output_dim = 19 , init = 'uniform', activation='relu', input_dim = 38))
ann.add(Dense(output_dim = 19, init = 'uniform', activation = 'relu'))
ann.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

ann.fit(X_train, Y_train, batch_size= 10 , nb_epoch = 100)

ann_pred = ann.predict(X_test)


ann.fit(X,Y, batch_size=10, nb_epoch = 100)
ann_pred1 = ann.predict(X1)

#Accuracy using ROC and AOC

from sklearn.metrics import roc_auc_score
log_auc = roc_auc_score(Y_test, lv_pred)
knn_auc = roc_auc_score(Y_test, knn_pred)
sv_auc = roc_auc_score(Y_test, sv_pred)
svc_auc = roc_auc_score(Y_test, svc_pred)
dc_auc = roc_auc_score(Y_test, dc_pred)
rf_auc = roc_auc_score(Y_test, rf_pred)
ann_auc = roc_auc_score(Y_test, ann_pred)




#grid search
ann_pred2 = []
for line in ann_pred1:
    for word in line:
        ann_pred2.append(word)
    
output= pd.DataFrame({'id' :X_id, 'is_pass' :ann_pred2})
output.to_csv('submission.csv', index = False)


