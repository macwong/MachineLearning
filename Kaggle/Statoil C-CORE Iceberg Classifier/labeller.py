# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 08:55:54 2017

@author: Dave
"""

import pandas as pd

class PseudoLabeller:
    def __init__(self, trainer):
        self.trainer = trainer
        
        if trainer.has_trained == False:
            trainer.train(32, 1, False)
        
        self.results = pd.DataFrame()
        
        for model in trainer.models:
            name = model.get_name()
            print(name)
            test = model.predict(trainer.X_test, False)
            self.results[name] = test[:, 0]
            
        print(self.results)