import abc
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import clone
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
        DaveModelBase.train_model(self.model_men, self.X_train_men, self.y_train_men)
        DaveModelBase.analyse(self.model_men, self.X_train_men, self.y_train_men)
        
        print("\nTraining women...")
        DaveModelBase.train_model(self.model_women, self.X_train_women, self.y_train_women)
        DaveModelBase.analyse(self.model_women, self.X_train_women, self.y_train_women)
        
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
        submission = pd.DataFrame()
        submission["submission_id"] = men_test_ids.append(women_test_ids)
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
        
        submission.to_csv("AUS_SubmissionFormat_LogisticRegression.csv", index=False)
    
    def save_model(self):
        pass
    
    def run(self):
        pass
    

class MyLogisticRegression(DaveModelBase):
    def get_model(self):
        return LogisticRegression()
    
    def get_name(self):
        return "Logistic Regression"