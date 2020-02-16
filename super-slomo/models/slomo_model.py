import tensorflow as tf
from tensorflow import keras as k

from models import layers


class SloMoNet(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(SloMoNet).__init__(name="SloMoNet", *args, **kwargs)
        self.frame_0 = k.Input(shape=(None,), name="frame_0")
        self.frame_t = k.Input(shape=(None,), name="frame_t")
        self.frame_1 = k.Input(shape=(None,), name="frame_1")
        self.flow_comp_layer = layers.UNet(4, name="flow_comp")
        self.optical_flow = layers.OpticalFlow(name="optical_flow")
        self.output_layer = layers.Output(name="predictions")
        self.warping_layer = layers.WarpingOutput(name="warping_output")
        self.t = 0

    def call(self, inputs, training=False, **kwargs):
        flow_input = tf.concat([self.frame_0, self.frame_1], axis=3)
        flow_out, flow_enc = self.flow_comp_layer(flow_input)
        f_01, f_t0, v_t0, f_10, f_t1, v_t1 = self.optical_flow(flow_out)
        predictions_input = [self.frame_0, f_t0, v_t0, self.frame_1, f_t1, v_t1]
        predictions = self.output_layer(predictions_input)
        warping_input = [self.frame_0, self.frame_1, f_01, f_10, f_t0, f_t1]
        warping_output = self.warping_layer(warping_input)
        return predictions, warping_output
