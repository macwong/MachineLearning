from sklearn.model_selection import train_test_split

class Trainer:
    def __init__(self, X, y, models):
        self.X = X
        self.y = y
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X, y, shuffle=False, test_size=0.20)
        self.models = models
        self.predict = []
        
    def train(self, batch_size = 32, epochs = 80, saveModel = True):
        for model in self.models:
            model.train(self.X_train, self.y_train, self.X_val, self.y_val, batch_size, epochs, saveModel)

    def train_full(self, batch_size = 32, epochs = 80, saveModel = True):
        for model in self.models:
            model.train(self.X_train, self.y_train, batch_size = batch_size, epochs = epochs, saveModel = saveModel)
                        
    def predict(self, X_test):
        self.predict = []

        for model in self.models:
            self.predict.append(model.predict(X_test))
            
    def submit(self):
        prediction = np.zeros((8424, ))
        
        for pred in self.predict:
            prediction += pred[:, 1]
            
        prediction = prediction / len(self.predict)
        
        submission = pd.DataFrame(test, columns=["id"])
        
        submission["is_iceberg"] = prediction

        test_func = lambda p: round(p["is_iceberg"], 4)
        submission["is_iceberg"] = test_func(submission)
        submission["is_iceberg"] = submission["is_iceberg"].round(4)
        submission.to_csv("submission.csv", float_format='%g', index = False)