# import the necessary packages
from tensorflow.keras.applications.resnet50 import ResNet50

class ResNet50Net:
    @staticmethod
    def build(size):
    
        model = ResNet50(weights="imagenet", include_top=False, input_shape=(size, size, 3))
    
        # don't train existing weights
        for layer in model.layers:
            layer.trainable = False
        
        model.summary()
        
        return model