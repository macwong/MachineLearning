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
            
        if trainer.has_predicted == False:
            trainer.predict(submit = False)
            
        self.predictions = trainer.predictions
        
        self.results = pd.DataFrame()
        
        for model in trainer.models:
            name = model.get_name()
            print(name)
            self.results[name] = model.predictions[:, 0]
            
        print(self.results)