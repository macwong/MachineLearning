import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score

class DaveModelBase:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = self.get_model()
        self.best_model = clone(self.model)
        
    def train(self):
        DaveModelBase.train_model(self.best_model, self.X_train, self.y_train)
        DaveModelBase.analyse(self.best_model, self.X_train, self.y_train)
        
    def train_model(model, X_train, y_train):
        print("\nTraining...")
        model.fit(X_train, y_train)

        
    def analyse(model, X_train, y_train):
        pred_train = model.predict_proba(X_train)
        pred_train = np.clip(pred_train, 0.0001, 0.9999)
        print("Log loss:", log_loss(y_train, pred_train))
        
    def predict(self):
        return DaveModelBase.predict_model(self.best_model, self.X_test)
    
    def predict_model(model, X_test):
        print("\nPredicting...")
        preds = model.predict_proba(X_test)
        print("Prediction count:", preds.shape)
        
        return preds       

    def evaluate_cv(self):
        model = clone(self.model)
        results = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='log_loss', n_jobs=4)
        print("({0:.3f}) +/- ({1:.3f})".format(results.mean(), results.std()))

    def evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X_train, self.y_train, test_size = 0.25, random_state = 42)
        model = clone(self.model)
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
    
    def create_submission(self):
        pass
    
    def save_model(self):
        pass
    
    def run(self):
        pass
    

class MyLogisticRegression(DaveModelBase):
    def get_model(self):
        return LogisticRegression()
    
    def get_name(self):
        return "Logistic Regression"