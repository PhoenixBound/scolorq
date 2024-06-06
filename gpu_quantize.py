import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import tensorflow as tf

def gumbel_softmax(logits, tau, hard):
    # Sample Gumbel(0, 1) distribution to get random noise
    u = keras.random.uniform(keras.ops.shape(logits))
    g = keras.ops.stop_gradient(-keras.ops.log(-keras.ops.log(u)))
    # Add noise to the log probabilities and softmax it
    x = (keras.ops.log(logits) + g) / tau
    result = keras.ops.softmax(x)
    
    if hard:
        # Assuming channels-last order
        max_indices = keras.ops.argmax(result, axis=-1)
        result_hard = keras.ops.one_hot(max_indices)
        # And add the gradients, using the trick mentioned in PyTorch's docs
        return result_hard - keras.ops.stop_gradient(result) + result
    else:
        return result

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
        return xSomething to note, the final rescue will be available on the 15th, its recommended that you do it ASAP and other Liz Requests as soon as you can, so you can prepare to do the secret boss of this game (they made it so that the boss does not require New Game Plus to do, unlike in the original/FES)

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

class Clip01Constraint(keras.constraints.Constraint):
    def __call__(self, w):
        return keras.ops.clip(w, 0., 1.)

initializer = GaussianBlurKernelInitializer((1.0, 0.3))
print(initializer((5, 5, 3, 2)))


class EdgePadding2D(keras.layers.Layer):
    def __init__(self, padding: (int, int)):
        super().__init__()
        self.symmetric_height_pad = padding[0]
        self.symmetric_width_pad = padding[1]

    def call(self, inputs):
        # Use 'symmetric' mode several times for better backend support: https://stackoverflow.com/a/50175380
        vertical_padding_added = 0
        horizontal_padding_added = 0
        while vertical_padding_added < self.symmetric_height_pad and \
                horizontal_padding_added < self.symmetric_width_pad:
            vpad = min(self.symmetric_height_pad - vertical_padding_added, vertical_padding_added + 1)
            hpad = min(self.symmetric_width_pad - horizontal_padding_added, horizontal_padding_added + 1)
            inputs = keras.ops.pad(inputs, ((0,0), (vpad, vpad), (hpad, hpad), (0,0)), mode='symmetric')
        return inputs



# Input: float tensor with dimensions [batch size, width, height, number of colors in palette]
#        This tensor represents the probability of each color being used for a part of the image.
# Output: float tensor with dimensions [batch size] describing loss values
#         This tensor represents how far off the image is.

# Sequence:
# * Take gumbel softmax of input tensor

# * Convert from (reparameterized) "one-hot encoded" palette indices to the raw color values
#   When tau = 0 for the gumbel softmax, this is equivalent to choosing a color from the palette.
palette_4bpp = keras.layers.Embedding(16, 3, embeddings_constraint=Clip01Constraint)
# Pad with same color at the edges
edge_paddig = EdgePadding2D((1, 1))
# Apply a blur
blur = keras.layers.DepthwiseConv2D((3, 3), padding='valid', data_format='channels_last', use_bias=False, depthwise_initializer=GaussianBlurKernelInitializer, trainable=False)
# Convert to a perceptually uniform color space (UCS)
to_ucs = LinearSrgbToOklab()
# Diff with original image in Oklab


