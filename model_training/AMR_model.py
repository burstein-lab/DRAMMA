from correlated_features import get_features_to_drop
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


class AMRModel:
    def __init__(self, X_train, y_train, selected_features=(), n_feats=0, n_jobs=2, param_dict=None):
        self.features = selected_features if len(selected_features) > 0 else AMRModel.get_best_model_features(X_train, y_train, n=n_feats, param_dict=param_dict, n_jobs=n_jobs)
        pipe = AMRModel.create_model_pipeline(n_jobs=n_jobs, param_dict=param_dict)
        self.model = pipe.fit(X_train[self.features], y_train.values.ravel())

    @staticmethod
    def create_model_pipeline(number_of_trees=0, n_jobs=1, model='rf', param_dict=None):
        tree_param = 'n_estimators' if model == 'rf' else 'num_parallel_tree'
        param_dict = param_dict if param_dict is not None else {}
        param_dict = {tree_param: number_of_trees} if len(param_dict) == 0 and number_of_trees > 0 else param_dict
        param_dict['random_state'] = 42
        if model == 'rf':
            clf = RandomForestClassifier(n_jobs=n_jobs, **param_dict)
        numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),('scaler', StandardScaler())])
        return Pipeline(steps=[('numeric_transformer', numeric_transformer), ("clf", clf)])

    @staticmethod
    def get_best_model_features(X_train, y_train, n, model_name='rf', n_trees=250, n_jobs=2, param_dict=None):
        """Trains a model using all no-null features to get feature importance, uses it to filter out
        correlated features, and then selects n features with the highest feature importance."""
        features = X_train.dropna(how='all', axis=1).columns  # no-null features
        pipe = AMRModel.create_model_pipeline(number_of_trees=n_trees, n_jobs=n_jobs, model=model_name, param_dict=param_dict)
        temp_model = pipe.fit(X_train[features], y_train.values.ravel())
        sorted_features = list(features[list(np.argsort(temp_model.steps[1][1].feature_importances_))[::-1]])
        to_drop = get_features_to_drop(X_train, sorted_features, 0.95)
        if n == 0:  # returns all non-correlated features
            return list(X_train.drop(to_drop, axis=1).columns)

        sorted_features = [feat for feat in sorted_features if feat not in to_drop]
        return sorted_features[:n]
