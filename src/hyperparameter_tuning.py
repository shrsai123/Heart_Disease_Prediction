from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier


def logistic_regression_hyperparam(X_train, y_train):
    logistic_param_grid = {
    'C':[0.001,0.01,0.1,1,10,100,1000],
    'penalty':['l1','l2'],
    'solver':['liblinear']
}
    model = GridSearchCV(LogisticRegression(max_iter=1000), logistic_param_grid, cv=5)
    model.fit(X_train, y_train)
    return model.best_params_, model.best_estimator_

def knn_hyperparam(X_train, y_train):
    knn_param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],  
    'weights': ['uniform', 'distance'],  
    'metric': ['euclidean', 'manhattan', 'minkowski']  
}
    model = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
    model.fit(X_train, y_train)
    return model.best_params_, model.best_estimator_


def random_forest_hyperparam(X_train, y_train):
    randfor_param_grid = {
    'n_estimators': [25, 30, 40, 50, 75, 100, 150, 200],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [8, 9, 10, 11, 12],
    'criterion' :['gini', 'entropy']
}
    model = GridSearchCV(RandomForestClassifier(random_state=42), randfor_param_grid, cv=5)
    model.fit(X_train, y_train)
    return model.best_params_, model.best_estimator_
    

def xgboost_hyperparam(X_train, y_train):
    xgboost_param_grid = {
    'n_estimators':[50,100,200],
    'learning_rate':[0.01,0.1,0.2],
    'max_depth':[3,4,5,7],
    'min_child_weight':[1,3,5],
    'subsample':[0.5,0.7,0.8,1.0],
    'colsample_bytree':[0.5,0.7,0.8,1.0]
}
    model = GridSearchCV(xgb.XGBClassifier(n_estimators=100, objective='binary:logistic', random_state=42), xgboost_param_grid, cv=5)
    model.fit(X_train, y_train)
    return model.best_params_, model.best_estimator_
