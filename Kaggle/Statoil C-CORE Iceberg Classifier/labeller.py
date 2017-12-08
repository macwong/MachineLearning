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
            trainer.train(32, 5, False)
            
        if trainer.has_predicted == False:
            trainer.predict(submit = False)
            
        self.predictions = trainer.predictions
        
        self.results = pd.DataFrame(columns = ['ids', 'definitive'])
        
        for model in trainer.models:
            name = model.get_name()
            print(name)
            self.results[name] = model.predictions[:, 0]
            
        self.results['ids'] = trainer.ids
        self.results['definitive'] = self.results.apply(lambda x: self.is_definitive(trainer.models, x), axis = 1)
        self.results["label"] = self.results.apply(lambda x: self.get_label(trainer.models, x), axis = 1)
        
        print(self.results)
        
    def is_definitive(self, models, result):
        is_pos = True
        is_neg = True
        
        for model in models:
            is_pos = is_pos == True and result[model.get_name()] > 0.5
            is_neg = is_neg == True and result[model.get_name()] <= 0.5
        
        return is_pos | is_neg
    
    def get_label(self, models, result):
        score = 0
        
        for model in models:
            if result[model.get_name()] > 0.5:
                score += 1
            else:
                score -= 1
            
        label = -1
        
        if score > 0:
            label = 1
        elif score < 0:
            label = 0
            
        return label