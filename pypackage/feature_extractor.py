# -*- coding: utf-8 -*-
# import the necessary packages

class FeatureExtractor:
    @staticmethod
    
    def extract(model, input_param):
    
        #Now, let us use features from convolutional network
        features = model.predict(input_param)
        
        features = features.reshape(features.shape[0], -1)
        
        return features
