from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import time
import datetime

class Trainer:
    def __init__(self, ids, X, y, X_test, models):
        self.has_trained = False
        self.has_predicted = False
        self.ids = ids
        self.X = X
        self.y = y
        self.X_test = X_test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, shuffle=False, test_size=0.20, random_state = 23)
        self.models = models
        self.predictions = []
        
    def train(self, batch_size = 32, epochs = 80, saveModel = True):
        for model in self.models:
            model.train(self.X_train, self.y_train, self.X_val, self.y_val, batch_size, epochs, saveModel)
            self.has_trained = True
            
    def train_full(self, batch_size = 32, epochs = 80, saveModel = True):
        for model in self.models:
            model.train(self.X, self.y, batch_size = batch_size, epochs = epochs, saveModel = saveModel)
            self.has_trained = True
            
    def predict(self, submit = True):
        self.predictions = []

        for model in self.models:
            self.predictions.append(model.predict(self.X_test, submit = False))
            
        if submit:
            self.submit()
            
        self.has_predicted = True
            
    def submit(self):
        prediction = np.zeros((8424, ))
        
        for pred in self.predictions:
            prediction += pred[:, 1]
            
        prediction = prediction / len(self.predictions)
        
        submission = pd.DataFrame(self.ids, columns=["id"])
        
        submission["is_iceberg"] = prediction

        test_func = lambda p: round(p["is_iceberg"], 4)
        submission["is_iceberg"] = test_func(submission)
        submission["is_iceberg"] = submission["is_iceberg"].round(4)
        submission.to_csv("submission" + datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H%M%S') + ".csv", float_format='%g', index = False)