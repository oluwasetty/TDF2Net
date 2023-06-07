# import the necessary packages
from tensorflow.keras.applications.vgg16 import VGG16

class VGG16Net:
    @staticmethod
    def build(size):
    
        model = VGG16(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    
        # don't train existing weights
        for layer in model.layers:
            layer.trainable = False
        
        model.summary()
        
        return model