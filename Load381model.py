from keras.models import load_model
from adabelief_tf import AdaBeliefOptimizer
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.layers import Layer, InputSpec


def loadMobileNetModel():
    name = load_model('mask_models/saved_models/mobileNetModel381.h5')
    return name


def loadOurOwnModel():
    name = load_model('mask_models/saved_models/ourOwnModel381fixed.h5')
    return name
