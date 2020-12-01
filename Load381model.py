from keras.models import load_model
from adabelief_tf import AdaBeliefOptimizer


def loadMobileNetModel():
    name = load_model('./mask_models/mobileNetModel381.h5')
    return name


def loadOurOwnModel():
    name = load_model('./mask_models/ourOwnModel381.h5')
    return name
