import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
 
from src.preprocessing import make_resampling_pipeline
 
 
def _strip_clf_prefix(params):
    return {k.replace("clf__", ""): v for k, v in params.items()}
 
 
def _run_grid_search(estimator, param_grid, X_train, y_train,
                     scoring="roc_auc", random_state=42, n_jobs=-1):
    pipeline = make_resampling_pipeline(estimator, random_state=random_state)
    # Re-key the grid to address the 'clf' step inside the pipeline
    pipe_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        pipeline,
        pipe_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    gs.fit(X_train, y_train)
    return _strip_clf_prefix(gs.best_params_), gs.best_estimator_
 
 

 
def logistic_regression_hyperparam(X_train, y_train, random_state=42):
    grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }
    estimator = LogisticRegression(max_iter=1000, random_state=random_state)
    return _run_grid_search(estimator, grid, X_train, y_train, random_state=random_state)
 
 
def knn_hyperparam(X_train, y_train, random_state=42):
    grid = {
        "n_neighbors": [3, 5, 7, 9, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan", "minkowski"],
    }
    estimator = KNeighborsClassifier()
    return _run_grid_search(estimator, grid, X_train, y_train, random_state=random_state)
 
 
def random_forest_hyperparam(X_train, y_train, random_state=42):
    grid = {
        "n_estimators": [25, 30, 40, 50, 75, 100, 150, 200],
        "max_features": ["sqrt", "log2"],
        "max_depth": [8, 9, 10, 11, 12],
        "criterion": ["gini", "entropy"],
    }
    estimator = RandomForestClassifier(random_state=random_state)
    return _run_grid_search(estimator, grid, X_train, y_train, random_state=random_state)
 
 
def xgboost_hyperparam(X_train, y_train, random_state=42):
    grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 4, 5, 7],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.5, 0.7, 0.8, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.8, 1.0],
    }
    estimator = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state
    )
    return _run_grid_search(estimator, grid, X_train, y_train, random_state=random_state)
 