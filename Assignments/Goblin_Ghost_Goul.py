import pandas as pd
import os
import io
from sklearn import model_selection
from sklearn import tree
from sklearn import ensemble
import numpy as np
from sklearn import preprocessing
os.chdir("D:/Data Science/Data/")

train=pd.read_csv("train.csv")
train.shape
train.info()

test=pd.read_csv("test.csv")
test.shape
test.info()
test['type']=None

train1=pd.concat([train,test])
train1.shape
train1.info()

train2=pd.get_dummies(train1,columns=['color'])
train2.shape
train3=train2.drop(['type'],axis=1,inplace=False)

X_train=train3[0:train.shape[0]]
X_train.shape

y_train=train['type']

dt_estimator=tree.DecisionTreeClassifier()
rf_estimator=ensemble.RandomForestClassifier()
ada_estimator=ensemble.AdaBoostClassifier()

voting_estimator=ensemble.VotingClassifier(estimators=[('dt',dt_estimator),('rf',rf_estimator),('ada',ada_estimator)],voting='soft',weights=[4,4,5])
voting_grid={'dt__max_depth':[3,5,7],'rf__n_estimators':[20,30],'rf__max_features':[7,8],'rf__max_depth':[7,8,9],'ada__n_estimators':[50]}
voting_grid_estimator=model_selection.GridSearchCV(voting_estimator,voting_grid,cv=10,n_jobs=5)
voting_grid_estimator.fit(X_train, y_train)

print(voting_grid_estimator.grid_scores_)
print(voting_grid_estimator.best_score_)
print(voting_grid_estimator.best_params_)
print(voting_grid_estimator.score(X_train,y_train))


x_test=train3[train.shape[0]:]
x_test.shape

test['type']=voting_grid_estimator.predict(x_test)

test.to_csv("Submission.csv",columns=['id','type'],index=False)
