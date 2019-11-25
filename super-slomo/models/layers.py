import tensorflow as tf


class UNet(tf.keras.Model):
    def __init__(self, out_filters, *args, **kwargs):
        super(UNet).__init__(name="UNet", *args, **kwargs)
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1,
                                            padding=3)
        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=1,
                                            padding=3)
        self.encoder1 = Encoder(64, 5)
        self.encoder2 = Encoder(128, 3)
        self.encoder3 = Encoder(256, 3)
        self.encoder4 = Encoder(512, 3)
        self.encoder4 = Encoder(512, 3)
        self.decoder1 = Decoder(512)
        self.decoder2 = Decoder(256)
        self.decoder3 = Decoder(128)
        self.decoder4 = Decoder(64)
        self.decoder5 = Decoder(32)
        self.conv3 = tf.keras.layers.Conv2D(filters=out_filters, kernel_size=3, strides=1,
                                            padding=1)

    def call(self, inputs, training=False, **kwargs):
        x = self.conv1(inputs)
        x = self.leaky_relu(x)
        skip = self.conv2(x)
        skip1 = self.leaky_relu(skip)
        skip2 = self.encoder1(skip1)
        skip3 = self.encoder2(skip2)
        skip4 = self.encoder3(skip3)
        skip5 = self.encoder4(skip4)
        x = self.encoder5(skip5)
        x = self.decoder1([x, skip5])
        x = self.decoder2([x, skip4])
        x = self.decoder3([x, skip3])
        x = self.decoder4([x, skip2])
        x = self.decoder5([x, skip1])
        x = self.conv3(x)
        x = self.leaky_relu(x)
        return x


class Encoder(tf.keras.layers.Layer):
    """
    Average Pooling -> CNN + Leaky ReLU -> CNN + Leaky ReLU
    Used to create a UNet like architecture.
    """
    def __init__(self, filters, kernel_size, **kwargs):
        super(Encoder).__init__(**kwargs)
        self.padding = (kernel_size - 1) // 2
        self.filters = filters
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding=self.padding)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=self.kernel_size, strides=1, padding=self.padding)
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
        super(Decoder).__init__(**kwargs)
        self.filters = filters

    def build(self, input_shape):
        self.conv1 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=self.filters, kernel_size=3, strides=1, padding=1)
        self.interpolation = tf.keras.layers.UpSampling2D(interpolation="bilinear")
        self.leaky_relu = tf.keras.layers.LeakyReLU(alpha=0.1)

    def call(self, inputs, **kwargs):
        x, skip = inputs
        x = self.interpolation(inputs)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        cat = tf.keras.layers.Concatenate(axis=1)([x, skip])
        cat = self.conv2(cat)
        cat = self.leaky_relu(cat)
        return cat



