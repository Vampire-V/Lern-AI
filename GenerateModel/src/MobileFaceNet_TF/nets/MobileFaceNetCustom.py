from collections import namedtuple
import tensorflow as tf
from tensorflow.keras import layers, regularizers, initializers, Model

# Conv and InvResBlock namedtuple define layers of the MobileNet architecture
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth', 'ratio'])
DepthwiseConv = namedtuple('DepthwiseConv', ['kernel', 'stride', 'depth', 'ratio'])
InvResBlock = namedtuple('InvResBlock', ['kernel', 'stride', 'depth', 'ratio', 'repeate'])

# _CONV_DEFS specifies the MobileNet body
_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=64, ratio=1),
    DepthwiseConv(kernel=[3, 3], stride=1, depth=64, ratio=1),

    InvResBlock(kernel=[3, 3], stride=2, depth=64, ratio=2, repeate=5),
    InvResBlock(kernel=[3, 3], stride=2, depth=128, ratio=4, repeate=1),
    InvResBlock(kernel=[3, 3], stride=1, depth=128, ratio=2, repeate=6),
    InvResBlock(kernel=[3, 3], stride=2, depth=128, ratio=4, repeate=1),
    InvResBlock(kernel=[3, 3], stride=1, depth=128, ratio=2, repeate=2),

    Conv(kernel=[1, 1], stride=1, depth=512, ratio=1),
]

# Define inverted block
def inverted_block(net, input_filters, output_filters, expand_ratio, stride, scope=None):
    '''Fundamental network structure of inverted residual block'''
    with tf.name_scope(scope):
        res_block = layers.Conv2D(input_filters * expand_ratio, (1, 1), activation=None)(net)
        res_block = layers.DepthwiseConv2D((3, 3), strides=stride, depth_multiplier=1, padding='same')(res_block)
        res_block = layers.Conv2D(output_filters, (1, 1), activation=None)(res_block)
        if stride == 2:
            return res_block
        else:
            if input_filters != output_filters:
                net = layers.Conv2D(output_filters, (1, 1), activation=None)(net)
            return tf.keras.layers.Add()([res_block, net])

# Define MobileNetV2 base structure
def mobilenet_v2_base(inputs, final_endpoint='Conv2d_7', min_depth=8, conv_defs=None, scope=None):
    depth = lambda d: max(int(d), min_depth)
    end_points = {}

    if conv_defs is None:
        conv_defs = _CONV_DEFS

    if not isinstance(conv_defs, (list, tuple)):
        raise ValueError("conv_defs must be a list or tuple of convolution definitions.")

    with tf.name_scope(scope or 'MobileFaceNetCustom'):
        net = inputs
        for i, conv_def in enumerate(conv_defs):
            end_point_base = f'Conv2d_{i}'

            if isinstance(conv_def, Conv):
                net = layers.Conv2D(depth(conv_def.depth), conv_def.kernel, strides=conv_def.stride,
                                    padding='same', kernel_initializer=initializers.GlorotUniform(),
                                    kernel_regularizer=regularizers.L2(0.00005))(net)
                end_points[end_point_base] = net
                if end_point_base == final_endpoint:
                    return net, end_points

            elif isinstance(conv_def, DepthwiseConv):
                net = layers.DepthwiseConv2D(conv_def.kernel, strides=conv_def.stride,
                                             padding='same', depthwise_initializer=initializers.GlorotUniform(),
                                             depthwise_regularizer=regularizers.L2(0.00005))(net)
                net = layers.Conv2D(conv_def.depth, (1, 1), activation=None)(net)
                end_points[end_point_base] = net
                if end_point_base == final_endpoint:
                    return net, end_points

            elif isinstance(conv_def, InvResBlock):
                input_filters = net.shape[-1]
                if input_filters is None:  # กรณี symbolic Tensor
                    input_filters = tf.keras.backend.int_shape(net)[-1]

                net = inverted_block(net, input_filters, depth(conv_def.depth), conv_def.ratio, conv_def.stride, end_point_base)
                for index in range(1, conv_def.repeate):
                    suffix = f'_{index}'
                    net = inverted_block(net, input_filters, depth(conv_def.depth), conv_def.ratio, 1, end_point_base + suffix)
                end_points[end_point_base] = net
                if end_point_base == final_endpoint:
                    return net, end_points
            else:
                raise ValueError(f'Unknown convolution type {type(conv_def)} for layer {i}')
    raise ValueError(f'Unknown final endpoint {final_endpoint}')

# Define MobileNetV2 model
def mobilenet_v2(inputs, bottleneck_layer_size=128, is_training=False, min_depth=8, conv_defs=None, global_pool=False, activation="relu"):
    # ตรวจสอบว่า TensorFlow ทำงานใน Eager Execution Mode
    assert tf.executing_eagerly(), "TensorFlow is not in Eager Execution mode!"

    # ตรวจสอบรูปร่างอินพุต
    input_shape = list(inputs.shape)
    if len(input_shape) != 4:
        raise ValueError(f'Invalid input tensor rank, expected 4, was: {len(input_shape)}')

    # เรียกใช้งาน mobilenet_v2_base
    net, end_points = mobilenet_v2_base(inputs, min_depth=min_depth, conv_defs=conv_defs)

    # Global Pooling (ลดมิติ Spatial)
    net = tf.keras.layers.GlobalAveragePooling2D()(net)

    # Bottleneck Layer
    net = layers.Dense(bottleneck_layer_size, activation=activation)(net)

    return net, end_points

# Define the MobileFaceNetCustom class
class MobileFaceNetCustom(Model):
    def __init__(self, embedding_size=128, **kwargs):
        super(MobileFaceNetCustom, self).__init__(**kwargs)
        self.embedding_size = embedding_size
        self.base_model = None

    def build(self, input_shape):
        inputs = tf.keras.Input(shape=(112, 112, 3), name="input_image")
        net, _ = mobilenet_v2(inputs, bottleneck_layer_size=self.embedding_size)
        self.base_model = tf.keras.Model(inputs=inputs, outputs=net, name="MobileFaceNetCustom")

    def call(self, inputs):
        return self.base_model(inputs)

    def get_config(self):
        config = super(MobileFaceNetCustom, self).get_config()
        config.update({"embedding_size": self.embedding_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


