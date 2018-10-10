"""
Beating the benchmark 
Otto Group product classification challenge @ Kaggle
@author: Fernandroid
"""

import pandas as pd
import numpy as np
from sklearn import ensemble, feature_extraction, preprocessing, grid_search, cross_validation

# import data
train = pd.read_csv('...\Otto/train.csv')
test = pd.read_csv('...\Otto/test.csv')
sample = pd.read_csv('...\Otto/sampleSubmission.csv')

train.dtypes
train.describe()
train.info()
test.info()
# drop ids and get labels
labels = train.target.values
train = train.drop('id', axis=1)
train = train.drop('target', axis=1)
train_data=train.values
test = test.drop('id', axis=1)
test=test.values[:,1:]

# transform counts to TFIDF features
tfidf = feature_extraction.text.TfidfTransformer()
train = tfidf.fit_transform(train).toarray()
test = tfidf.transform(test).toarray()

# encode labels 
lbl_enc = preprocessing.LabelEncoder()
labels = lbl_enc.fit_transform(labels)

#Split data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data,labels,test_size=0.2, random_state=0)
from sklearn.externals import joblib
joblib.dump(X_train, '...\Otto_group\X_train.pkl')
joblib.dump(y_train, '...\Otto_group\y_train.pkl')
joblib.dump(X_test, '...\Otto_group\X_test.pkl')
joblib.dump(y_test, '...\Otto_group\y_test.pkl')

X_train=joblib.load('...\Otto_group\X_train.pkl')
y_train=joblib.load('...\Otto_group\y_train.pkl')
X_test=joblib.load('...\Otto_group\X_test.pkl')
y_test=joblib.load('...\Otto_group\y_test.pkl')

#Standardize data
scaler = preprocessing.StandardScaler().fit(X_train)
mean=scaler.mean_
scaler.std_
X_train=scaler.transform(X_train)
joblib.dump(scaler, '...\Otto_group\scaler.pkl')
scaler=joblib.load('...\Otto_group\scaler.pkl')
#train SVC classifier
#search hyperparameter
from sklearn import svm
svc = svm.SVC(kernel='linear')
parameters = {'C':[0.5,1,10,100]}

svc = svm.SVC(kernel='rbf',probability=True,C=100,gamma=1)
svc.fit(X_train,y_train)
svc.scores(X_train,y_train)
parameters = {'C':[10,100],'gamma':[1]}
clf = grid_search.GridSearchCV(svc, parameters,cv=3,verbose=5)
clf.fit(X_train,y_train)
report= clf.grid_scores_
clf.best_estimator_
param=clf.best_params_
clf.predict(X_test,y_test)
#training model
svc=svm.SVC(kernel='rbf',probability=True).set_params(**param)
svc.fit(X_train, y_train).score(X_test,y_test)
pred=svc.predict(X_test)
prob=svc.predict_proba(X_test)

pred=svc.predict(test)
prob=svc.predict_proba(test)
# train a random forest classifier
clf = ensemble.RandomForestClassifier(n_estimators=500,criterion='gini',verbose=5)
clf.fit(X_train, y_train)
pred=clf.predict_proba(X_test)
from sklearn.metrics import log_loss
log_loss(y_test, pred)
"""
350  0.5979
450  0.584
"""
" AdaBoost "
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
ada=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(criterion='entropy',max_depth=8,max_features=0.5),
                       n_estimators=400,learning_rate=0.5)
ada.fit(X_train, y_train)
ada.score(X_train,y_train)
P=ada.predict_proba(X_test)
multiclass_log_loss(y_test, P)
from sklearn.metrics import log_loss
log_loss(y_real, y_pred)

"""
estimators max_depth score test_logloss
100        3         58% 
100        8         76%
250        8         78%       0.818
350        8         77%
learning rate 0.5
250        8         82.88%  
300        8         83.5%     1.04  
"""
# Gradient Boosting model
X=np.vstack([X_train,X_test])
Y=np.hstack([y_train,y_test])
from sklearn.ensemble import GradientBoostingClassifier
boo = GradientBoostingClassifier(n_estimators=520,max_depth=8,learning_rate=0.05,subsample=1,max_features=0.2,min_samples_leaf=70 ,verbose=5)
boo.fit(X_train,y_train)
boo.fit(X,Y)
P=boo.predict_proba(X_test)
from sklearn.metrics import log_loss
log_loss(y_test, P)
"""
estimators max_depth  Lr   subsample  max_features  test_logloss
130          8        0.1    0.6        0.2           0.5149 xxx
130          8        0.1    0.7        0.2           0.50483 xxxx 
130          8        0.1    0.8        0.2           0.49583 xxxxx
130          8        0.1    0.9        0.2           0.49324 xxxxx
490          8       0.05    1          0.2  minsamplesleaf 50 0.4690
490          8       0.04    1          0.2  minsamplesleaf 50 0.46746 xxx
500          8       0.05    1          0.2  minsamplesleaf 60 0.46771
520          8       0.05    1          0.2  minsamplesleaf 60 0.46687 xxxxx
520          8       0.06    1          0.2  minsamplesleaf 65 0.47078 
"""
joblib.dump(boo, '...\Otto_group\GBC.pkl') 
joblib.dump(boo, '...\Otto_group\GBC_2.pkl') 

""" Neuronal Network """
runfile('.../Otto/DropoutNN2hl.py', wdir=r'.../Otto')
#runfile('.../Otto/NNSoftMax.py', wdir=r'.../Otto')
input_layer_size=93
hidden_layer_size=200
num_labels=9
landa=1
P=np.array([[1,0.5]])
Theta1=np.random.rand(hidden_layer_size,input_layer_size+1)*2*(1./np.sqrt(input_layer_size+1))-(1./np.sqrt(input_layer_size+1))
Theta2=np.random.rand(num_labels,hidden_layer_size+1)*2*(1./np.sqrt(hidden_layer_size+1))-(1./np.sqrt(hidden_layer_size+1))
nn_params=np.concatenate((Theta1.flatten(),Theta2.flatten()))
J, grad= DropoutNN2hl(nn_params,P,input_layer_size,hidden_layer_size,num_labels,X_train, y_train,landa)

batch=100
Lr=0.1
for iter in xrange(100):
for i in xrange(0,X_train.shape[0]-batch,batch):
    X=X_train[i:i+batch,:]
    Y=y_train[i:i+batch]
    for j in xrange(0,batch):
        Xb=X[j]
        Yb=Y[j]
        J, grad= DropoutNN2hl(nn_params,P,input_layer_size,hidden_layer_size,num_labels,Xb, Yb,landa)
        if j==0:
            gradM=grad
            cost=J
        else:
            gradM +=grad
            cost +=J
    gradM=gradM/batch
    cost=cost/batch
    #Lr=0.1/(1+iter*0.1)
    #nn_params=nn_params-Lr*grad
    nn_params-=Lr*gradM

from scipy.optimize import fmin_bfgs
J, grad= NNSoftMax(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train, y_train,landa)
def decorated_cost(nn_params):
    return NNSoftMax(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train, y_train,landa)[0]
def decorated_grad(nn_params):
    return NNSoftMax(nn_params,input_layer_size,hidden_layer_size,num_labels,X_train, y_train,landa)[1] 

sol=fmin_bfgs(decorated_cost,nn_params,fprime=decorated_grad,  maxiter=1,disp=1)

Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size+1 )], (hidden_layer_size, input_layer_size + 1))
Theta2 = np.reshape(nn_params[(hidden_layer_size * (input_layer_size + 1)):],(num_labels, hidden_layer_size + 1))
m = X_test.shape[0]
X = np.hstack([np.ones((m, 1)), X_test])
z2=np.dot(X,Theta1.T)
a2=sigmoid(z2)
a2 = np.hstack([np.ones((m, 1)), a2])
z3=np.dot(a2,Theta2.T)
P=SoftMax(z3)


boo=joblib.load('...\Otto_group\GBCtotale11.pkl')
# predict on test set
preds = clf.predict_proba(test)
preds = boo.predict_proba(test)
# create submission file
preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
preds.to_csv('...Otto_group/ModelBoo.csv', index_label='id')
preds.to_csv('...\Otto_group/ModelBoototal.csv', index_label='id')
preds.to_csv('...\Otto_group/ModelBoototal3.csv', index_label='id')
preds.to_csv('...\Otto_group/ModelBoo5.csv', index_label='id')
preds.to_csv('...\Otto_group/ModelBoototal11.csv', index_label='id')

