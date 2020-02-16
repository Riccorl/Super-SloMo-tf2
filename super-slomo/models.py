import tensorflow as tf
from tensorflow import keras as k
import tensorflow_addons as tfa

def get_model():
    k.backend.clear_session()

    frame_0 = k.Input(shape=(None,), name="frame_0")
    frame_1 = k.Input(shape=(None,), name="frame_1")
