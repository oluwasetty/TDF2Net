# import the necessary packages
from tensorflow.keras.applications.xception import Xception, preprocess_input

class XceptionNet:
    @staticmethod
    def build(size):
    
        model = Xception(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    
        # don't train existing weights
        for layer in model.layers:
            layer.trainable = False
        
        model.summary()
        
        return model
    
    def preprocess(inp):
        
        #preprocess the input         
        inp = preprocess_input(inp)
        
        return inp