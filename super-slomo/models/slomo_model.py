import numpy as np
import tensorflow as tf

from models import layers


class SloMoNet(tf.keras.Model):
    def __init__(self, n_frames=12, name="SloMoNet", **kwargs):
        super(SloMoNet, self).__init__(name=name, **kwargs)
        t_slices = tf.constant(np.linspace(0, 1, n_frames))
        self.flow_comp_layer = layers.UNet(4, name="flow_comp")
        self.optical_flow = layers.OpticalFlow(t=t_slices, name="optical_flow")
        self.output_layer = layers.Output(t=t_slices, name="predictions")
        self.warping_layer = layers.WarpingOutput(name="warping_output")

    def call(self, inputs, training=False, **kwargs):
        frames_0, frames_1, frames_i = inputs
        flow_input = tf.concat([frames_0, frames_1], axis=3)
        flow_out = self.flow_comp_layer(flow_input)
        optical_input = [frames_0, frames_1, flow_out, frames_i]
        f_01, f_t0, v_t0, f_10, f_t1, v_t1 = self.optical_flow(optical_input)
        preds_input = [
            frames_0,
            f_t0,
            v_t0,
            frames_1,
            f_t1,
            v_t1,
            frames_i,
        ]
        predictions = self.output_layer(preds_input)
        warping_input = [frames_0, frames_1, f_01, f_10, f_t0, f_t1]
        warping_output = self.warping_layer(warping_input)
        losses_output = [f_01, f_10, f_t0, f_t1] + warping_output
        return predictions, losses_output
