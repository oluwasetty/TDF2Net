# import the necessary packages
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

class InceptionV3Net:
    @staticmethod
    def build(size):
    
        model = InceptionV3(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    
        # don't train existing weights
        for layer in model.layers:
            layer.trainable = False
        
        model.summary()
        
        return model
    
    def preprocess(inp):
        
        #preprocess the input         
        inp = preprocess_input(inp)
        
        return inp