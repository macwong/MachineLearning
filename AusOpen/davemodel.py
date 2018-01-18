import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import clone
from sklearn.model_selection import train_test_split

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
        
    def train_model(model, X_train, y_train):
        print("\nTraining...")
        model.fit(X_train, y_train)
        
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
#    
#    def predict()
#    
#    def evaluate(data, truth):
#        X_train, X_test, y_train, y_test = train_test_split(data, truth, test_size = 0.25, random_state = 42)
#        log_reg_test, pred_test = predict(X_train, y_train, X_test) 
#        analyse(log_reg_test, X_test, y_test)
#    
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
    

class MyLogisticRegression(DaveModelBase):
    def get_model(self):
        return LogisticRegression()
    
    def get_name(self):
        return "Logistic Regression"