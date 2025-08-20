# Importing Required packages
import json as json
import numpy as np
import pandas as pd
import warnings

from striprtf.striprtf import rtf_to_text
import time
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction import FeatureHasher
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score,make_scorer
from sklearn.exceptions import DataConversionWarning
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor,ExtraTreesClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings(action="ignore", category=DataConversionWarning)
# ------------------------------------------------
# Custom Transformer classes for Feature Selection 
# ------------------------------------------------
class NoReductionSelection(BaseEstimator, TransformerMixin):
  '''
  This transformer performs no feature reduction
  '''
  def fit(self, X, y=None):
    self.X = X
    return self

  def transform(self, X, *_):
    return self.X

class CorrWithTargetSelection(BaseEstimator, TransformerMixin):
  '''
  This transformer performs feature selection based on colrrelation
  of features with the target. The features with highest absolute
  correlation with the target variable are selected.
  '''
  def __init__(self, n_features: int):
    self.n_features = n_features
    self.selected_columns = None

  def fit(self, X, y = None):
    combined_Xy = np.hstack((X,y)).astype(np.float32)
    correlations = np.corrcoef(combined_Xy, rowvar=False)
    self.selected_columns = np.argsort(np.abs(correlations[-1][:-1]))[- self.n_features:]
    return self

  def transform(self, X, y=None, **kwargs):
    return X[ : , self.selected_columns]

  def get_params(self, deep=False):
    return {"n_features": self.n_features}
  
class TreeBasedSelection(BaseEstimator, TransformerMixin):
  '''
  This transformer performs feature selection based on feature importances generated
  by a tree based estimator algorithm.
  (`RandomForestRegressor` or `RandomForestClassifier` for this class.)
  '''
  def __init__(self, n_features: int, n_trees: int = 5, depth: int = 10, task_type: str = 'Regression'):
    self.n_features = n_features
    self.n_trees = n_trees
    self.depth = depth
    self.task_type = task_type
    self.estm = None

  def fit(self, X, y = None):
    if self.task_type == 'Regression':
      self.estm = RandomForestRegressor(n_estimators = self.n_trees,
                                             max_depth = self.depth,
                                             random_state = 42)
    else:
      self.estm = RandomForestClassifier(n_estimators = self.n_trees,
                                              max_depth = self.depth,
                                              random_state = 42)

    self.estm.fit(X,y)
    return self

  def transform(self, X, y=None, **kwargs):
    return SelectFromModel(self.estm, prefit=True,
                           max_features=self.n_features,
                           threshold = -np.inf).transform(X)

  def get_params(self, deep=False):
    return {"n_features": self.n_features, "n_trees": self.n_trees, "depth":self.depth}
  
# ----------------------------------------
# Creating the Build Models class
# ----------------------------------------
class BuildModels:
    def __init__(self, param_path, data_path):
        self.allParams = self._convertRtfToDict(param_path)
        self.rmse = make_scorer(self._rmse_loss, greater_is_better=False)
        self.dataPath = data_path

        self.feature_handling_params = self.allParams['design_state_data']['feature_handling']
        self.algorithm_params = self.allParams['design_state_data']['algorithms']
        self.feature_reduction_params = self.allParams['design_state_data']['feature_reduction']
        self.target_params = self.allParams['design_state_data']['target']

        self.predType = str(self.target_params['prediction_type'])
        self.targetFeature = self.target_params['target']
        self.featuresUsed = [feature for feature in self.feature_handling_params if self.feature_handling_params[feature]['is_selected']]
        self.algorithmsUsed = [algo for algo in self.algorithm_params if self.algorithm_params[algo]['is_selected']]
        (self.X_train, self.X_test, self.y_train, self.y_test), self.featurePositions = self._loadAndSplitDataset(self.dataPath,self.targetFeature,self.featuresUsed)
        self.estimatorTypes = { "RandomForestClassifier":"Classification",
                                "GBTClassifier":"Classification",
                                "LogisticRegression":"Classification",
                                "DecisionTreeClassifier":"Classification",
                                "SVM":"Classification",
                                "KNN":"Classification",
                                "extra_random_trees":"Classification",
                                "neural_network":"Classification",
                                "RandomForestRegressor":"Regression",
                                "GBTRegressor":"Regression",
                                "LinearRegression":"Regression",
                                "RidgeRegression":"Regression",
                                "LassoRegression":"Regression",
                                "ElasticNetRegression":"Regression",
                                "DecisionTreeRegressor":"Regression"}
    
    def taskInformation(self):
        print("Prediction type:",self.predType)
        print("Features:",self.featuresUsed)
        print("Target:",self.targetFeature)
        print("Algorithms used:",self.algorithmsUsed)
    
    def buildAndTestModels(self) -> dict:
      trainedModels = {}
      for index,algorithm in enumerate(self.algorithmsUsed):
        if(algorithm in self.estimatorTypes.keys()):
          if(self.estimatorTypes[algorithm] == self.predType):
            try:
              start = time.time()
              model,training_perf = self._buildBestEstimatorPipeline(algorithm,self.X_train,self.y_train)
              stop = time.time() 
              y_pred_test = model.predict(self.X_test)
              if(self.predType == "Regression"):
                test_perf = {"R2-Score":r2_score(self.y_test, y_pred_test),"RMSE":self._rmse_loss(self.y_test, y_pred_test)}
              else:
                test_perf = {"Accuracy":accuracy_score(self.y_test,y_pred_test)}
              print(f'[{index + 1}] Algorithm: {algorithm}\nTraining performance :\n{training_perf}\n\nTest performance :\n{test_perf}\n')
              print(f"Best estimator hyperparameters obtained for {algorithm}:\n{model.named_steps['Estimator']}\n", end = "\n")
              print(f'Training time (Seconds): {stop - start}\n', end="-----" * 16 + "\n")
              trainedModels[algorithm] = {'trained_model':model,'training_time(S)':stop - start,'training_perf':training_perf,'test_perf':test_perf}
            except NotFittedError:
               print(f'\n[{index + 1}] {algorithm} Failed to fit\n', end = '-----' * 16 + '\n')
          else:
            print(f'\n[{index + 1}] {algorithm} is invalid model type for {self.predType}\n', end = '-----' * 16 + '\n')
        else:
          print(f'\n[{index + 1}] {algorithm} is currently undefined!\n', end = '-----' * 16 + '\n')      
      
      return trainedModels
    
    def _buildBestEstimatorPipeline(self, estimatorName: str, X_train, y_train):
      # ----------------------------------------------------------
      # Common Parameters for Pipelines
      # ----------------------------------------------------------
      imputeEncodeTransformer = ColumnTransformer(self._createImputersAndEncoders(self.featuresUsed,self.targetFeature), remainder='passthrough')
      featureReductionTransformer = self._featureReducer()
      params = self.algorithm_params[estimatorName]
      training_metrics_regression = {"R2-Score":None,"RMSE":None}
      training_metrics_classification = {"Accuracy":None}
      # ----------------------------------------------------------
      # GridSearch and Pipeline Creation of various estimators
      # ----------------------------------------------------------
      if(estimatorName == "DecisionTreeRegressor"):
        min_depth = int(params['min_depth'])
        max_depth = int(params['max_depth'])
        min_samples_leaf = params['min_samples_per_leaf']
        split = []
        if params['use_best']:
                split.append('best')

        if params['use_random']:
                split.append('random')

        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', DecisionTreeRegressor())])
        param_grid = {'Estimator__splitter':split,
                      'Estimator__max_depth': range(min_depth, max_depth + 1),
                      'Estimator__min_samples_leaf': min_samples_leaf}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression

      elif(estimatorName == "ElasticNetRegression"):
        min_iter = int(params['min_iter'])
        max_iter = int(params['max_iter'])
        min_regp = float(params['min_regparam'])
        max_regp = float(params['max_regparam'])
        min_enet = float(params['min_elasticnet'])
        max_enet = float(params['max_elasticnet'])
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', ElasticNet())])
        param_grid = {'Estimator__max_iter': range(min_iter, max_iter + 1),
                      'Estimator__l1_ratio': np.arange(min_enet, max_enet, 0.1),
                      'Estimator__alpha': np.arange(min_regp, max_regp, 0.1)}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression


      elif(estimatorName == "LassoRegression"):
        min_iter = int(params['min_iter'])
        max_iter = int(params['max_iter'])
        min_regp = float(params['min_regparam'])
        max_regp = float(params['max_regparam'])
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', Lasso())])
        param_grid = {'Estimator__max_iter': range(min_iter, max_iter + 1),
                      'Estimator__alpha': np.arange(min_regp, max_regp, 0.1)}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression

      elif(estimatorName == "RidgeRegression"):
        min_iter = int(params['min_iter'])
        max_iter = int(params['max_iter'])
        min_regp = float(params['min_regparam'])
        max_regp = float(params['max_regparam'])
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', Ridge())])
        param_grid = {'Estimator__max_iter': range(min_iter, max_iter + 1),
                      'Estimator__alpha': np.arange(min_regp, max_regp, 0.1)}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression

      elif(estimatorName == "LinearRegression"):
        jobs = -1
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', LinearRegression())])
        param_grid = {'Estimator__n_jobs': [jobs]}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression

      elif(estimatorName == "GBTRegressor"):
        min_depth = int(params['min_depth'])
        max_depth = int(params['max_depth'])
        n_trees = params['num_of_BoostingStages']
        if params['feature_sampling_statergy'] == "Fixed number":
          max_features = [int(params['fixed_number'])]
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', GradientBoostingRegressor())])
        param_grid = {'Estimator__n_estimators': n_trees,
                      'Estimator__max_depth': range(min_depth, max_depth + 1),
                      'Estimator__max_features': max_features}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression

      elif(estimatorName == "RandomForestRegressor"):
        min_depth = int(params['min_depth'])
        max_depth = int(params['max_depth'])
        min_trees = int(params['min_trees'])
        max_trees = int(params['max_trees'])
        min_samples_leaf = range(int(params['min_samples_per_leaf_min_value']),int(params['min_samples_per_leaf_max_value']) + 1)
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', RandomForestRegressor())])
        param_grid = {'Estimator__n_estimators': range(min_trees, max_trees + 1),
                      'Estimator__max_depth': range(min_depth, max_depth + 1),
                      'Estimator__min_samples_leaf': min_samples_leaf,
                      'Estimator__n_jobs': [-1]}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = self.rmse, cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_regression["R2-Score"] = r2_score(y_train,y_pred_train)
        training_metrics_regression["RMSE"] = self._rmse_loss(y_train,y_pred_train)
        return best_pipeline,training_metrics_regression

      elif(estimatorName == "neural_network"):
        hidden_layer_sizes = params['hidden_layer_sizes']
        alpha = [float(params['alpha_value'])]
        beta_1 = [float(params['beta_1'])]
        beta_2 = [float(params['beta_2'])]
        momentum = [float(params['momentum'])]
        max_iter = [int(params['max_iterations'])]
        shuffle = [params['shuffle_data']]
        tol = [float(params['convergence_tolerance'])]
        lr_init = [float(params['initial_learning_rate'])]
        power_t = [float(params['power_t'])]
        early_stopping = [params['early_stopping']]
        nesterovs_momentum = [params['use_nesterov_momentum']]

        if(str(params['activation']).lower() in ['identity','logistic','tanh','relu']):
          activation = [str(params['activation']).lower()]
        else:
          activation = ['relu']

        if(str(params['solver']).lower() in ['lbfgs','sgd','adam']):
          solver = [str(params['solver']).lower()]
        else:
          solver = ['adam']

        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', MLPClassifier())])
        param_grid = {'Estimator__hidden_layer_sizes': hidden_layer_sizes,
                      'Estimator__alpha': alpha,
                      'Estimator__beta_1': beta_1,
                      'Estimator__beta_2': beta_2,
                      'Estimator__momentum': momentum,
                      'Estimator__max_iter': max_iter,
                      'Estimator__shuffle': shuffle,
                      'Estimator__tol': tol,
                      'Estimator__learning_rate_init': lr_init,
                      'Estimator__power_t': power_t,
                      'Estimator__early_stopping': early_stopping,
                      'Estimator__nesterovs_momentum': nesterovs_momentum,
                      'Estimator__activation': activation,
                      'Estimator__solver': solver}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "extra_random_trees"):
        n_estimators = params['num_of_trees']
        max_depth = params['max_depth']
        min_samples_leaf = params['min_samples_per_leaf']
        max_features = [None]

        if("square" in params['feature_sampling_statergy'].lower()):
          max_features.append("sqrt")
          if(None in max_features):
            max_features.remove(None)

        if("log" in params['feature_sampling_statergy'].lower()):
          max_features.append("log2")
          if(None in max_features):
            max_features.remove(None)

        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', ExtraTreesClassifier())])
        param_grid = {'Estimator__n_estimators': n_estimators,
                      'Estimator__max_depth': max_depth,
                      'Estimator__min_samples_leaf': min_samples_leaf,
                      'Estimator__max_features': max_features,
                      'Estimator__n_jobs': [-1]}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "KNN"):
        n_neighbors = params['k_value']
        if(params['distance_weighting']):
          weights = ['distance']
        else:
          weights = ['uniform']
        p_value = [int(params['p_value'])]
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', KNeighborsClassifier())])
        param_grid = {'Estimator__n_neighbors': n_neighbors,
                      'Estimator__weights': weights,
                      'Estimator__p': p_value}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "SVM"):
        c_value = params['c_value']
        tol = [float(params['tolerance'])]
        max_iter = [int(params['max_iterations'])]

        kernel = []
        if(params['linear_kernel']):
          kernel.append('linear')
        if(params['polynomial_kernel']):
          kernel.append('poly')
        if(params['rep_kernel']):
          kernel.append('rbf')
        if(params['sigmoid_kernel']):
          kernel.append('sigmoid')
        if(len(kernel) == 0):
          kernel.append('rbf')

        gamma = []
        if(params['auto']):
          gamma.append('auto')
        if(params['scale']):
          gamma.append('scale')
        if(len(gamma) == 0):
          gamma.append('scale')

        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', SVC())])
        param_grid = {'Estimator__C': c_value,
                      'Estimator__tol': tol,
                      'Estimator__max_iter': max_iter,
                      'Estimator__kernel': kernel,
                      'Estimator__gamma': gamma}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "DecisionTreeClassifier"):
      
        if (not(params['use_gini']) and params['use_entropy']):
          criterion = 'entropy'
        elif(params['use_gini'] and not(params['use_entropy'])):
          criterion = 'gini'

        min_depth = int(params['min_depth'])
        max_depth = int(params['max_depth'])
        min_samples_leaf = params['min_samples_per_leaf']
        split = []
        if params['use_best']:
                split.append('best')

        if params['use_random']:
                split.append('random')

        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', DecisionTreeClassifier())])
        param_grid = {'Estimator__splitter':split,
                      'Estimator__criterion': [criterion],
                      'Estimator__max_depth': range(min_depth, max_depth + 1),
                      'Estimator__min_samples_leaf': min_samples_leaf}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "LogisticRegression"):
        min_iter = int(params['min_iter'])
        max_iter = int(params['max_iter'])
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', LogisticRegression())])
        param_grid = {'Estimator__max_iter': range(min_iter, max_iter + 1),
                      'Estimator__n_jobs': [-1]}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "GBTClassifier"):
        min_depth = int(params['min_depth'])
        max_depth = int(params['max_depth'])
        n_trees = params['num_of_BoostingStages']
        if params['feature_sampling_statergy'] == "Fixed number":
          max_features = [int(params['fixed_number'])]
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', GradientBoostingClassifier())])
        param_grid = {'Estimator__n_estimators': n_trees,
                      'Estimator__max_depth': range(min_depth, max_depth + 1),
                      'Estimator__max_features': max_features}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1)
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification

      elif(estimatorName == "RandomForestClassifier"):
        min_depth = int(params['min_depth'])
        max_depth = int(params['max_depth'])
        min_trees = int(params['min_trees'])
        max_trees = int(params['max_trees'])
        min_samples_leaf = range(int(params['min_samples_per_leaf_min_value']),int(params['min_samples_per_leaf_max_value']) + 1)
        pipe = Pipeline(steps = [ ('ImputerEncoder', imputeEncodeTransformer),
                                  ('FeatureReducer', featureReductionTransformer),
                                  ('Estimator', RandomForestClassifier())])
        param_grid = {'Estimator__n_estimators': range(min_trees, max_trees + 1),
                      'Estimator__max_depth': range(min_depth, max_depth + 1),
                      'Estimator__min_samples_leaf': min_samples_leaf,
                      'Estimator__n_jobs': [-1]}
        grid_search = GridSearchCV(estimator = pipe, param_grid = param_grid,
                                   scoring = 'accuracy', cv = 2, n_jobs = -1 )
        grid_search.fit(X_train, y_train)
        best_pipeline = grid_search.best_estimator_
        y_pred_train = best_pipeline.predict(X_train)
        training_metrics_classification["Accuracy"] = accuracy_score(y_train,y_pred_train)
        return best_pipeline,training_metrics_classification
      
    def _featureReducer(self):
      params = self.feature_reduction_params
      strategy = params['feature_reduction_method']
      if(strategy == "No Reduction"):
        return NoReductionSelection()
      elif(strategy == "Correlation with target"):
        n_features = int(params['num_of_features_to_keep'])
        return CorrWithTargetSelection(n_features = n_features)
      elif(strategy == "Tree-based"):
        n_features = int(params['num_of_features_to_keep'])
        n_trees = int(params['num_of_trees'])
        depth = int(params['depth_of_trees'])
        return TreeBasedSelection(n_features = n_features, n_trees = n_trees, depth = depth, task_type = self.predType)
      elif(strategy == "Principal Component Analysis"):
        n_features = int(params['num_of_features_to_keep'])
        return PCA(n_components = n_features, random_state = 42)  
    
    def _createImputersAndEncoders(self,features :list, target :str):
        if(target in features):
            features.remove(target)
        transformerList = []
        for feature in features:
            feature_handling = self.feature_handling_params[feature]
            if(feature_handling['feature_variable_type'] == "numerical"):
                if(feature_handling['feature_details']['missing_values'] == "Impute"):
                    if(feature_handling['feature_details']['impute_with'] == "Average of values"):
                        imputer = SimpleImputer(strategy = 'mean')
                        transformerList.append((f'{feature}_imputer',imputer,[self.featurePositions[feature]]))
                    elif(feature_handling['feature_details']['impute_with'] == "custom"):
                        imputer = SimpleImputer(strategy = 'constant', fill_value = feature_handling['feature_details']['impute_value'])
                        transformerList.append((f'{feature}_imputer',imputer,[self.featurePositions[feature]]))
            else:
                if(feature_handling['feature_details']['text_handling'] == "Tokenize and hash"):
                    encoder = FeatureHasher(n_features = 2, input_type="string")
                    transformerList.append((f'{feature}_encoder',encoder,[self.featurePositions[feature]]))

        return transformerList

    def _rmse_loss(self,y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true,y_pred))

    def _convertRtfToDict(self,path: str) -> dict:
        param_file = open(path,'r')
        param_file_content = param_file.read()
        param_file_content = rtf_to_text(param_file_content)
        param_file.close()
        return json.loads(param_file_content)
    
    def _loadAndSplitDataset(self, path: str, target: str, features: list, val_split: float = 0.25, random_state : int = 42):
        df = pd.read_csv(path)
        y = df[[target]].values.reshape(-1,1)
        if(target in features):
            features.remove(target)
        
        if(self.predType == 'Classification'):
          y = LabelEncoder().fit_transform(y) 
        X_df = df.drop([target], axis = 1)[features]
        featurePositions = {val:index for index,val in enumerate(X_df.columns)}
        X = X_df.values
        return train_test_split(X, y, test_size = val_split, random_state = random_state), featurePositions

# -----------------------   
# Putting it all together
# -----------------------
if __name__ == '__main__':
  ALGOPARAMS_PATH_ORIGINAL = 'original_task_params/algoparams_from_ui.json.rtf'
  ALGOPARAMS_PATH_MODIFIED = 'modified_params_for_testing/algoparams_from_ui.json.rtf'
  DATA_PATH = 'iris.csv'
  # ------------------------------------------------- 
  # Use Modified params to test classification models
  # -------------------------------------------------
  ModelBuilder = BuildModels(param_path=ALGOPARAMS_PATH_ORIGINAL,data_path=DATA_PATH)
  print("Some preliminary information:")
  ModelBuilder.taskInformation()
  print("-----"*16+"\nBuilding and Testing models:\n")
  trainedModels = ModelBuilder.buildAndTestModels()
  # ------------------------------------------------------
  # We can store the trained estimator pipelines if needed
  # ------------------------------------------------------
  
## Updated Line for test purposes