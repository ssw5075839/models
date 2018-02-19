# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""SSDFeatureExtractor for vgg features."""
import tensorflow as tf

from object_detection.meta_architectures import ssd_meta_arch
from object_detection.models import feature_map_generators
from object_detection.utils import ops
from object_detection.utils import shape_utils
from nets import vgg, custom_layers

slim = tf.contrib.slim


class SSDVgg16FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):
  """SSD Feature Extractor using vgg features."""

  def __init__(self,
               is_training,
               depth_multiplier,
               min_depth,
               pad_to_multiple,
               conv_hyperparams,
               batch_norm_trainable=True,
               reuse_weights=None,
               use_explicit_padding=False,
               use_depthwise=False):
    """vgg Feature Extractor for SSD Models.

    Args:
      is_training: whether the network is in training mode.
      depth_multiplier: float depth multiplier for feature extractor.
      min_depth: minimum feature extractor depth.
      pad_to_multiple: the nearest multiple to zero pad the input height and
        width dimensions to.
      conv_hyperparams: tf slim arg_scope for conv2d and separable_conv2d ops.
      batch_norm_trainable: Whether to update batch norm parameters during
        training or not. When training with a small batch size
        (e.g. 1), it is desirable to disable batch norm update and use
        pretrained batch norm params.
      reuse_weights: Whether to reuse variables. Default is None.
      use_explicit_padding: Whether to use explicit padding when extracting
        features. Default is False.
      use_depthwise: Whether to use depthwise convolutions. Default is False.
    """
    super(SSDVgg16FeatureExtractor, self).__init__(
        is_training, depth_multiplier, min_depth, pad_to_multiple,
        conv_hyperparams, batch_norm_trainable, reuse_weights,
        use_explicit_padding, use_depthwise)

  def preprocess(self, resized_inputs):
    """SSD preprocessing.

    Maps pixel values to the range [-1, 1].

    Args:
      resized_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.
    """
    return (2.0 / 255.0) * resized_inputs - 1.0

  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': ['vgg_16/conv4_3_norm', 'vgg_16/fc7', 'vgg_16/conv6_2', 'vgg_16/conv7_2', 'vgg_16/conv8_2', 'vgg_16/conv9_2'],
        'layer_depth': [-1, -1, -1, -1, -1, -1],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }
    net,image_features = vgg.vgg_16(ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
                                        final_endpoint='pool5',spatial_squeeze=False)
    #double check scale filler
    image_features['vgg_16/conv4_3_norm']=custom_layers.l2_normalization(image_features['vgg_16/conv4/conv4_3'],scaling=True,scope='vgg_16/conv4_3_norm')
    with slim.arg_scope(self._conv_hyperparams):
      with tf.variable_scope('vgg_16', reuse=self._reuse_weights) as scope:
        # In [5]: net
        # Out[5]: <tf.Tensor 'vgg_16/pool5/MaxPool:0' shape=(32, 18, 18, 512) dtype=float32>

        # In [6]: end_points
        # Out[6]:
        # OrderedDict([('vgg_16/conv1/conv1_1',
        #               <tf.Tensor 'vgg_16/conv1/conv1_1/Relu:0' shape=(32, 300, 300, 64) dtype=float32>),
        #              ('vgg_16/conv1/conv1_2',
        #               <tf.Tensor 'vgg_16/conv1/conv1_2/Relu:0' shape=(32, 300, 300, 64) dtype=float32>),
        #              ('vgg_16/pool1',
        #               <tf.Tensor 'vgg_16/pool1/MaxPool:0' shape=(32, 150, 150, 64) dtype=float32>),
        #              ('vgg_16/conv2/conv2_1',
        #               <tf.Tensor 'vgg_16/conv2/conv2_1/Relu:0' shape=(32, 150, 150, 128) dtype=float32>),
        #              ('vgg_16/conv2/conv2_2',
        #               <tf.Tensor 'vgg_16/conv2/conv2_2/Relu:0' shape=(32, 150, 150, 128) dtype=float32>),
        #              ('vgg_16/pool2',
        #               <tf.Tensor 'vgg_16/pool2/MaxPool:0' shape=(32, 75, 75, 128) dtype=float32>),
        #              ('vgg_16/conv3/conv3_1',
        #               <tf.Tensor 'vgg_16/conv3/conv3_1/Relu:0' shape=(32, 75, 75, 256) dtype=float32>),
        #              ('vgg_16/conv3/conv3_2',
        #               <tf.Tensor 'vgg_16/conv3/conv3_2/Relu:0' shape=(32, 75, 75, 256) dtype=float32>),
        #              ('vgg_16/conv3/conv3_3',
        #               <tf.Tensor 'vgg_16/conv3/conv3_3/Relu:0' shape=(32, 75, 75, 256) dtype=float32>),
        #              ('vgg_16/pool3',
        #               <tf.Tensor 'vgg_16/pool3/MaxPool:0' shape=(32, 37, 37, 256) dtype=float32>),
        #              ('vgg_16/conv4/conv4_1',
        #               <tf.Tensor 'vgg_16/conv4/conv4_1/Relu:0' shape=(32, 37, 37, 512) dtype=float32>),
        #              ('vgg_16/conv4/conv4_2',
        #               <tf.Tensor 'vgg_16/conv4/conv4_2/Relu:0' shape=(32, 37, 37, 512) dtype=float32>),
        #              ('vgg_16/conv4/conv4_3',
        #               <tf.Tensor 'vgg_16/conv4/conv4_3/Relu:0' shape=(32, 37, 37, 512) dtype=float32>),
        #              ('vgg_16/pool4',
        #               <tf.Tensor 'vgg_16/pool4/MaxPool:0' shape=(32, 18, 18, 512) dtype=float32>),
        #              ('vgg_16/conv5/conv5_1',
        #               <tf.Tensor 'vgg_16/conv5/conv5_1/Relu:0' shape=(32, 18, 18, 512) dtype=float32>),
        #              ('vgg_16/conv5/conv5_2',
        #               <tf.Tensor 'vgg_16/conv5/conv5_2/Relu:0' shape=(32, 18, 18, 512) dtype=float32>),
        #              ('vgg_16/conv5/conv5_3',
        #               <tf.Tensor 'vgg_16/conv5/conv5_3/Relu:0' shape=(32, 18, 18, 512) dtype=float32>),
        #              ('vgg_16/pool5',
        #               <tf.Tensor 'vgg_16/pool5/MaxPool:0' shape=(32, 18, 18, 512) dtype=float32>)])
        end_points_collection = scope.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):

          net = slim.convolution(net, 1024, [3, 3], padding='SAME', rate=6, scope='fc6')

          # def convolution(inputs,
          #         num_outputs,
          #         kernel_size,
          #         stride=1,
          #         padding='SAME',
          #         data_format=None,
          #         rate=1,
          #         activation_fn=nn.relu,
          #         normalizer_fn=None,
          #         normalizer_params=None,
          #         weights_initializer=initializers.xavier_initializer(),
          #         weights_regularizer=None,
          #         biases_initializer=init_ops.zeros_initializer(),
          #         biases_regularizer=None,
          #         reuse=None,
          #         variables_collections=None,
          #         outputs_collections=None,
          #         trainable=True,
          #         scope=None):

          #fc6 is dilated conv
          # layer {
          # name: "fc6"
          # type: "Convolution"
          # bottom: "pool5"
          # top: "fc6"
          #   param {
          #     lr_mult: 1.0
          #     decay_mult: 1.0
          #   }
          #   param {
          #     lr_mult: 2.0
          #     decay_mult: 0.0
          #   }
          #   convolution_param {
          #     num_output: 1024
          #     pad: 6
          #     kernel_size: 3
          #     weight_filler {
          #       type: "xavier"
          #     }
          #     bias_filler {
          #       type: "constant"
          #       value: 0.0
          #     }
          #     dilation: 6
          #   }
          # }
          # layer {
          #   name: "relu6"
          #   type: "ReLU"
          #   bottom: "fc6"
          #   top: "fc6"
          # }

          #fc7 is 1*1 conv
          # layer {
          #   name: "fc7"
          #   type: "Convolution"
          #   bottom: "fc6"
          #   top: "fc7"
          #   param {
          #     lr_mult: 1.0
          #     decay_mult: 1.0
          #   }
          #   param {
          #     lr_mult: 2.0
          #     decay_mult: 0.0
          #   }
          #   convolution_param {
          #     num_output: 1024
          #     kernel_size: 1
          #     weight_filler {
          #       type: "xavier"
          #     }
          #     bias_filler {
          #       type: "constant"
          #       value: 0.0
          #     }
          #   }
          # }
          # layer {
          #   name: "relu7"
          #   type: "ReLU"
          #   bottom: "fc7"
          #   top: "fc7"
          # }
          net = slim.conv2d(net,1024,[1,1],padding='SAME',scope='fc7')
          net = slim.conv2d(net,256, [1,1],padding='SAME',scope='conv6_1')
          net = slim.conv2d(net,512, [3,3],padding='SAME',stride=2,scope='conv6_2')
          net = slim.conv2d(net,128, [1,1],padding='SAME',scope='conv7_1')
          net = slim.conv2d(net,256, [3,3],padding='SAME',stride=2,scope='conv7_2')
          net = slim.conv2d(net,128, [1,1],padding='VALID',scope='conv8_1')
          net = slim.conv2d(net,256, [3,3],padding='VALID',stride=1,scope='conv8_2')
          net = slim.conv2d(net,128, [1,1],padding='VALID',scope='conv9_1')
          net = slim.conv2d(net,256, [3,3],padding='VALID',stride=1,scope='conv9_2')
          _image_features_new = slim.utils.convert_collection_to_dict(end_points_collection)

          for k,v in _image_features_new.iteritems():
            image_features[k] = v
        # _, image_features = inception_v3.inception_v3_base(
        #     ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
        #     final_endpoint='Mixed_7c',
        #     min_depth=self._min_depth,
        #     depth_multiplier=self._depth_multiplier,
        #     scope=scope)
    feature_maps = feature_map_generators.multi_resolution_feature_maps(
        feature_map_layout=feature_map_layout,
        depth_multiplier=self._depth_multiplier,
        min_depth=self._min_depth,
        insert_1x1_conv=True,
        image_features=image_features)

    return feature_maps.values()
