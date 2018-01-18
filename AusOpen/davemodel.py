import abc
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, cross_val_score

class DaveModelBase:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, ids_men, X_train_men, y_train_men, X_test_men, ids_women, X_train_women, y_train_women, X_test_women):
        self.ids_men = ids_men
        self.X_train_men = X_train_men
        self.y_train_men = y_train_men
        self.X_test_men = X_test_men
        self.model_men = self.get_model()

        self.ids_women = ids_women
        self.X_train_women = X_train_women
        self.y_train_women = y_train_women
        self.X_test_women = X_test_women
        self.model_women = self.get_model()

    def train(self):
        print("\nTraining men...")
        model_men = self.model_men
        model_women = self.model_women
        params = self.get_params()
        
        if params is not None:
            print(params)
            model_men = GridSearchCV(self.model_men, params, cv=5)
            model_women = GridSearchCV(self.model_women, params, cv=5)

        DaveModelBase.train_model(model_men, self.X_train_men, self.y_train_men)
        DaveModelBase.analyse(model_men, self.X_train_men, self.y_train_men)
        
        print("\nTraining women...")
        DaveModelBase.train_model(model_women, self.X_train_women, self.y_train_women)
        DaveModelBase.analyse(model_women, self.X_train_women, self.y_train_women)
        
    def train_model(model, X_train, y_train):
        model.fit(X_train, y_train)

        
    def analyse(model, X_train, y_train):
        pred_train = model.predict_proba(X_train)
        pred_train = np.clip(pred_train, 0.0001, 0.9999)
        print("Log loss:", log_loss(y_train, pred_train))
        
    def predict(self):
        print("\nPredicting men...")
        pred_men = DaveModelBase.predict_model(self.model_men, self.X_test_men)
        
        print("\nPredicting women...")
        pred_women = DaveModelBase.predict_model(self.model_women, self.X_test_women)
        
        return pred_men, pred_women
    
    def predict_model(model, X_test):
        preds = model.predict_proba(X_test)
        print("Prediction count:", preds.shape)
        
        return preds       

    def evaluate_cv(self):
        DaveModelBase.evaluate_cv_model(self.get_model(), self.X_train_men, self.y_train_men)
        DaveModelBase.evaluate_cv_model(self.get_model(), self.X_train_women, self.y_train_women)
        
    def evaluate_cv_model(model, X_train, y_train):
        results = cross_val_score(model, X_train, y_train, cv=5, scoring='log_loss', n_jobs=4)
        print("\nLog Loss: ({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

    def evaluate(self):
        DaveModelBase.evaluate_model(self.get_model(), self.X_train_men, self.y_train_men)
        DaveModelBase.evaluate_model(self.get_model(), self.X_train_women, self.y_train_women)

    def evaluate_model(model, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
        DaveModelBase.train_model(model, X_train, y_train)
        DaveModelBase.analyse(model, X_test, y_test)

    @abc.abstractmethod
    def get_model(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass
    
    def get_params(self):
        return None
    
    def create_submission(self, pred_men, pred_women):
        print("Creating submission...")
        submission = pd.DataFrame()
        submission["submission_id"] = self.ids_men.append(self.ids_women)
        submission["train"] = 0
        
        pred_men = np.clip(pred_men, 0.0001, 0.9999)
        pred_women = np.clip(pred_women, 0.0001, 0.9999)
        
        submission["UE"] = np.append(pred_men[:, 1], pred_women[:, 1], axis = 0)
        submission["FE"] = np.append(pred_men[:, 0], pred_women[:, 0], axis = 0)
        submission["W"] = np.append(pred_men[:, 2], pred_women[:, 2], axis = 0)
        
        submission_test = pd.read_csv("_RawData/AUS_SubmissionFormat.csv")
        sorter = submission_test["submission_id"]
        submission["submission_id"] = submission["submission_id"].astype("category")
        submission["submission_id"].cat.set_categories(sorter, inplace = True)
        submission = submission.sort_values(["submission_id"])
        
        submission.to_csv("AUS_SubmissionFormat_" + self.get_name() + ".csv", index=False)
        print("Done!")
        
    def save_model(self):
        with open(self.get_name() + '.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Saved")
    
    def load_model(file_name):
        with open(file_name, 'rb') as handle:
            b = pickle.load(handle)
            
        print("Loaded")
        
        return b
    
    def run(self):
        self.train()
        pred_men, pred_women = self.predict()
        self.create_submission(pred_men, pred_women)
        self.save_model()
    

class MyLogisticRegression(DaveModelBase):
    def get_model(self):
        return LogisticRegression()
    
    def get_name(self):
        return "LogisticRegression"
    
    def get_params(self):
        return {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'solver': ['newton-cg', 'sag', 'saga', 'lbfgs', 'liblinear']
        }
    

class MyRandomForest(DaveModelBase):
    def get_model(self):
        clf = RandomForestClassifier(random_state=42)
    
        param_grid = {
#                          'n_estimators': [45, 50, 60],
#                          'max_depth': [9, 11, 13, 15]
                         'n_estimators': [60],
                         'max_depth': [13],
                         'max_features': [10, 20]
                     }
    
        grid_clf = GridSearchCV(clf, param_grid, cv=10)
#        grid_clf.fit(X, y)
        
#        return RandomForestClassifier(n_estimators=100, max_features=10, random_state=23)
        
        return grid_clf
    
    def get_name(self):
        return "RandomForestClassifier"
    
    
class MyXGBoost(DaveModelBase):
    def get_model(self):
        return XGBClassifier(max_depth=7, learning_rate=0.012, n_estimators=1000, subsample=0.62, colsample_bytree=0.6, seed=1, n_jobs=4)
    
    def get_name(self):
        return "XGBoost"
    
class MyBagging(DaveModelBase):
    def get_model(self):
        return BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=23)
    
    def get_name(self):
        return "BaggingClassifier"

class MyGradientBoosting(DaveModelBase):
    def get_model(self):
        return GradientBoostingClassifier(n_estimators=100, max_features=10)
    
    def get_name(self):
        return "GradientBoostingClassifier"
    
    