import tensorflow as tf
import tensorflow_addons as ta


class UNet(tf.keras.layers.Layer):
    def __init__(self, out_filters, name="UNet", **kwargs):
        super(UNet, self).__init__(name=name, **kwargs)
        self.out_filters = out_filters

    def build(self, input_shape):
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding="same"
        )
        self.encoder1 = Encoder(32, 5)
        self.encoder2 = Encoder(64, 3)
        self.encoder3 = Encoder(128, 3)
        self.encoder4 = Encoder(256, 3)
        self.encoder5 = Encoder(256, 3)
        self.decoder1 = Decoder(256)
        self.decoder2 = Decoder(256)
        self.decoder3 = Decoder(256)
        self.decoder4 = Decoder(128)
        self.decoder5 = Decoder(64)
        self.conv3 = tf.keras.layers.Conv2D(
            filters=self.out_filters, kernel_size=3, strides=1, padding="same"
        )

    def call(self, inputs, **kwargs):
        x_enc = self.conv1(inputs)
        x_enc = self.leaky_relu(x_enc)
        skip = self.conv2(x_enc)
        skip1 = self.leaky_relu(skip)
        skip2 = self.encoder1(skip1)
        skip3 = self.encoder2(skip2)
        skip4 = self.encoder3(skip3)
        skip5 = self.encoder4(skip4)
        x_enc = self.encoder5(skip5)
        x_dec = self.decoder1([x_enc, skip5])
        x_dec = self.decoder2([x_dec, skip4])
        x_dec = self.decoder3([x_dec, skip3])
        x_dec = self.decoder4([x_dec, skip2])
        x_dec = self.decoder5([x_dec, skip1])
        x_dec = self.conv3(x_dec)
        x_dec = self.leaky_relu(x_dec)
        return x_dec, x_enc


class Encoder(tf.keras.layers.Layer):
    """
    Average Pooling -> CNN + Leaky ReLU -> CNN + Leaky ReLU
    Used to create a UNet like architecture.
    """

    def __init__(self, filters, kernel_size, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # self.padding = (kernel_size - 1) // 2
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
        )
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, **kwargs):
        x = self.avg_pool(inputs)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Bilinear interpolation -> CNN + Leaky ReLU -> CNN + Leaky ReLU
    Used to create a UNet like architecture.
    """

    def __init__(self, filters, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=3, strides=1, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.filters, kernel_size=3, strides=1, padding="same"
        )
        self.interpolation = tf.keras.layers.UpSampling2D(interpolation="bilinear")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, **kwargs):
        x, skip = inputs
        x = self.interpolation(x)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        cat = tf.keras.layers.Concatenate(axis=1)([x, skip])
        cat = self.conv2(cat)
        cat = self.leaky_relu(cat)
        return cat


class BackWarp(tf.keras.layers.Layer):
    """
    Backwarping to an image.
    Generate I_0 <- backwarp(F_0_1, I_1) given optical flow from frame I_0 to I_1 -> F_0_1 and frame I_1.
    """

    def __init__(self, **kwargs):
        super(BackWarp).__init__(**kwargs)

    def build(self, input_shape):
        self.backwarp = ta.image.dense_image_warp

    def call(self, inputs, **kwargs):
        image, flow = inputs
        img_backwarp = self.backwarp(image, flow)
        return img_backwarp


class OpticalFlow(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(OpticalFlow, self).__init__(**kwargs)
        self.t = 0.5

    def build(self, input_shape):
        self.flow_interp_layer = UNet(5, name="flow_interp")
        self.backwarp_layer_t0 = BackWarp()
        self.backwarp_layer_t1 = BackWarp()

    def call(self, inputs, **kwargs):
        # flow computation
        f_01, f_10 = inputs[:, 2:, :, :], inputs[:, :2, :, :]
        f_t0 = tf.add(
            tf.multiply((-1 * (1 - self.t) * self.t), f_01),
            tf.multiply(self.t * self.t, f_10),
        )
        f_t1 = ((1 - self.t) * (1 - self.t) * f_01) - (self.t * (1 - self.t) * f_10)
        # flow interpolation
        g_i0_ft0 = self.backwarp_layer_t0(self.frame_0, f_t0)
        g_i1_ft1 = self.backwarp_layer_t1(self.frame_1, f_t1)
        flow_interp_in = tf.concat(
            [self.frame_0, self.frame_1, f_01, f_10, f_t1, f_t0, g_i1_ft1, g_i0_ft0],
            axis=1,
        )

        flow_interp_out = self.flow_interp_layer(flow_interp_in)

        # optical flow residuals and visibility maps
        delta_f_t0 = flow_interp_out[:, :2, :, :]
        delta_f_t1 = flow_interp_out[:, 2:4, :, :]

        v_t0 = tf.keras.activations.sigmoid(flow_interp_out[:, 4:5, :, :])
        # v_t0 = tf.tile(v_t_0, [1, 1, 1, 3])
        v_t1 = 1 - v_t0

        f_t0 = tf.add(f_t0, delta_f_t0)
        f_t1 = tf.add(f_t1, delta_f_t1)
        return f_01, f_t0, v_t0, f_10, f_t1, v_t1


class Output(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Output, self).__init__(**kwargs)

    def build(self, input_shape):
        self.backwarp_layer_t0 = BackWarp()
        self.backwarp_layer_t1 = BackWarp()

    def call(self, inputs, **kwargs):
        frame_0, f_t0, v_t0, frame_1, f_t1, v_t1 = inputs

        z = tf.add(
            (tf.multiply(1 - self.t, v_t0), tf.add(tf.multiply(self.t * v_t1), 1e-12))
        )
        normalization_factor = tf.divide(1, z)

        frame_pred = tf.multiply(
            (1 - self.t) * v_t0, self.backwarp_layer_t0(frame_0, f_t0)
        ) + tf.multiply(self.t * v_t1, self.backwarp_layer_t1(frame_1, f_t1))
        frame_pred = tf.multiply(normalization_factor, frame_pred)
        return frame_pred


class WarpingOutput(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(WarpingOutput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.backwarp_layer1 = BackWarp()
        self.backwarp_layer2 = BackWarp()
        self.backwarp_layer3 = BackWarp()
        self.backwarp_layer4 = BackWarp()

    def call(self, inputs, **kwargs):
        frame_0, frame_1, f_01, f_10, f_t0, f_t1 = inputs

        return [
            self.backwarp_layer1(frame_1, f_01),
            self.backwarp_layer2(frame_0, f_10),
            self.backwarp_layer3(frame_0, f_t0),
            self.backwarp_layer4(frame_1, f_t1),
        ]


#
#
# class VGG_19(tf.keras.Model):
#     def __init__(self, **kwargs):
#         super(VGG_19).__init__(**kwargs)
#         self.scope = "vgg_19"
#         self.reuse = False
#
#     def call(self, input, **kwargs):
#         sc = tf.Variable(self.scope, self.scope, [input], reuse=self.reuse)
#         epc = sc.name + "_end_points"
#
#         tf.keras.layers.Conv2D
