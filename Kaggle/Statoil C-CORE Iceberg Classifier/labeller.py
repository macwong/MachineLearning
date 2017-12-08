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
        
        self.results = pd.DataFrame(columns = ['ids', 'definitive'])
        
        for model in trainer.models:
            name = model.get_name()
            print(name)
            self.results[name] = model.predictions[:, 0]
            
        self.results['ids'] = trainer.ids
        self.results['definitive'] = self.results.apply(lambda x: self.is_definitive(trainer.models, x), axis = 1)
#        very_positive =self.results.apply(lambda x: x["davemodel"] > 0.5 and x["vgg"] > 0.5 and x["vgg19"] > 0.5 and x["lenet"] > 0.5, axis = 1)
#        very_negative = self.results.apply(lambda x: x["davemodel"] <= 0.5 and x["vgg"] <= 0.5 and x["vgg19"] <= 0.5 and x["lenet"] <= 0.5, axis = 1)
#        self.results['definitive'] = very_positive | very_negative
            
#        self.results["very_positive"] = self.results.apply(lambda x: x["davemodel"] > 0.5 and x["vgg"] > 0.5 and x["vgg19"] > 0.5 and x["lenet"] > 0.5, axis = 1)
#        self.results["very_negative"] = self.results.apply(lambda x: x["davemodel"] <= 0.5 and x["vgg"] <= 0.5 and x["vgg19"] <= 0.5 and x["lenet"] <= 0.5, axis = 1)
#        self.results["definitive"] = self.results["very_positive"] | self.results["very_negative"]
            
        print(self.results)
        
    def is_definitive(self, models, result):
        is_pos = True
        is_neg = True
        
        for model in models:
            is_pos = is_pos == True and result[model.get_name()] > 0.5
            is_neg = is_neg == True and result[model.get_name()] <= 0.5
        
        return is_pos | is_neg