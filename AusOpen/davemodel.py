import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

class DaveModelBase:
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, X_train, y_train, X_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = self.get_model()
        
    def train(self):
        model = self.model
        model.fit(self.X_train, self.y_train)
        
        pred_train = model.predict_proba(self.X_train)
        pred_train = np.clip(pred_train, 0.0001, 0.9999)
        print("Log loss:", log_loss(self.y_train, pred_train))
    
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