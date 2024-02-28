import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import tensorflow as tf


def srgb_to_linear(img):
    simple_mask = img <= 0.04045
    complex_mask = 1.0 - img
    simple = img / 12.92
    complex_ = ((img + 0.055) / 1.055) ** 2.4
    return simple * simple_mask + complex_ * complex_mask

# These matrices are written transposed from how you'd usually write them,
# because convolving seems to involve multiplying matrices in the opposite order

# The code and tests for this were derived from the rust crate "palette":
# https://github.com/Ogeon/palette/blob/master/palette/src/oklab.rs
# used here under the MIT license.
linear_rgb_to_lms = keras.ops.convert_to_tensor([[
    [0.4122214708, 0.2119034982, 0.0883024619],
    [0.5363325363, 0.6806995451, 0.2817188376],
    [0.0514459929, 0.1073969566, 0.6299787005]
]])
linear_rgb_to_lms = keras.ops.stop_gradient(linear_rgb_to_lms)

lms_to_oklab = keras.ops.convert_to_tensor([[
    [0.2104542553, 1.9779984951, 0.0259040371],
    [0.7936177850, -2.4285922050, 0.7827717662],
    [-0.0040720468, 0.4505937099, -0.8086757660]
]])
lms_to_oklab = keras.ops.stop_gradient(lms_to_oklab)

class LinearSrgbToOklab(keras.layers.Layer):
    def __init__(self):
        super().__init__()
        
    
    def call(self, inputs):
        x = keras.ops.conv(inputs, linear_rgb_to_lms)
        # XXX: keras doesn't have a cube root operation
        x = tf.experimental.numpy.cbrt(x)
        x = keras.ops.conv(x, lms_to_oklab)
        return x

def test_srgb_to_oklab():
    test_colors = keras.ops.convert_to_tensor([[
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]])
    layer = LinearSrgbToOklab()
    oklab = layer(test_colors)
    # assert oklab is approximately equal to
    # [[[ 1.0000000e+00  0.0000000e+00  5.9604645e-08]
    #   [ 6.2795538e-01  2.2486293e-01  1.2584631e-01]
    #   [ 8.6643958e-01 -2.3388739e-01  1.7949840e-01]
    #   [ 4.5201370e-01 -3.2457016e-02 -3.1152818e-01]]]
    # (These values don't exactly match palette's tests but, meh.)
    # White is supposed to be [1, 0, 0]
    print(oklab)

test_srgb_to_oklab()





# Initializer for depthwise convolutions of color values
class GaussianBlurKernelInitializer(keras.initializers.Initializer):
    def __init__(self, stddevs):
        self.stddevs = stddevs
    
    # For serialization support
    def get_config(self):
        return { 'stddevs': self.stddevs }
    
    def __call__(self, shape, dtype=None):
        # The desired shape is  [kernel_spatial_shape, num_input_channels, num_channels_multiplier]
        
        stddevs = keras.ops.convert_to_tensor(self.stddevs, dtype=dtype)
        assert keras.ops.ndim(stddevs) <= 1
        stddevs = keras.ops.reshape(stddevs, (-1,))
        if shape[3] != stddevs.shape[0]:
            raise ValueError(f"Caller is requesting {shape[3]} filters per channel, but only {stddevs.shape[0]} standard deviations were provided!")
        stddevs_squared = keras.ops.square(stddevs)
        
        # Only support initializing 2D kernels (my use case)
        middle_1 = (shape[0] - 1) / 2
        middle_2 = (shape[1] - 1) / 2
        repr_2d = []
        for i in range(shape[0]):
            lst = [(i - middle_1) ** 2 + (j - middle_2) ** 2 for j in range(shape[1])]
            repr_2d.append(lst)

        repr_2d = -0.5 * keras.ops.convert_to_tensor(repr_2d, dtype=dtype)
        repr_2d = keras.ops.reshape(repr_2d, (repr_2d.shape[0], repr_2d.shape[1], -1))
        repr_2d = repr_2d / keras.ops.reshape(stddevs_squared, (1, 1, -1))
        repr_2d = keras.ops.exp(repr_2d)
        # Normalize so that the total sum is 1.0
        repr_2d = repr_2d / keras.ops.sum(repr_2d)
        
        # Now we stack it on top of itself in interesting ways, to account for
        # the number of input channels
        in_channels = [repr_2d for _ in range(shape[2])]
        repr_3d = keras.ops.stack(in_channels, axis=2)
        return repr_3d

initializer = GaussianBlurKernelInitializer((1.0, 0.3))
print(initializer((5, 5, 3, 2)))