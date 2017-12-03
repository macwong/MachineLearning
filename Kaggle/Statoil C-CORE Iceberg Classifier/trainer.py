class Trainer:
    def __init__(self, models):
        self.models = models
        self.predict = []
        
    def train(self, batch_size = 32, epoch = 80, saveMe = True):
        for model in self.models:
            model.train(batch_size, epoch, saveMe)
            
    def predict(self):
        self.predict = []

        for model in self.models:
            self.predict.append(model.predict())
            
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