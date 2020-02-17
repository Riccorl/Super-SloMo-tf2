import tensorflow as tf
from tensorflow import keras as k

from models import layers


class SloMoNet(tf.keras.Model):
    def __init__(self, name="SloMoNet", **kwargs):
        super(SloMoNet, self).__init__(name=name, **kwargs)
        self.flow_comp_layer = layers.UNet(4, name="flow_comp")
        self.optical_flow = layers.OpticalFlow(t=0.5, name="optical_flow")
        self.output_layer = layers.Output(name="predictions")
        self.warping_layer = layers.WarpingOutput(name="warping_output")

    def call(self, inputs, training=False, **kwargs):
        frame_0, frame_t, frame_1 = inputs
        flow_input = tf.concat([frame_0, frame_1], axis=3)
        flow_out, flow_enc = self.flow_comp_layer(flow_input)
        optical_input = [frame_0, frame_1, flow_out]
        f_01, f_t0, v_t0, f_10, f_t1, v_t1 = self.optical_flow(optical_input)
        predictions_input = [frame_0, f_t0, v_t0, frame_1, f_t1, v_t1]
        predictions = self.output_layer(predictions_input)
        warping_input = [frame_0, frame_1, f_01, f_10, f_t0, f_t1]
        warping_output = self.warping_layer(warping_input)
        return predictions, warping_output
