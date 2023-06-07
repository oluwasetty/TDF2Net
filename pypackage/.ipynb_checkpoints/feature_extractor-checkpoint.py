# -*- coding: utf-8 -*-
# import the necessary packages

class FeatureExtractor:
    @staticmethod
    
    def extract(model, input_param):
    
        #Now, let us use features from convolutional network
        feature_extractor=model.predict(input_param)
        
        features = feature_extractor.reshape(feature_extractor.shape[0], -1)
        
        return features
