from typing import Union, Optional, Tuple

import tensorflow as tf
from keras import Model
from keras.layers import (Activation, Add, BatchNormalization, Conv2D, Dropout,
                          GlobalAveragePooling2D, GlobalMaxPooling2D)
from keras.utils import get_file, get_source_inputs

S = Union[int, Tuple[int, int]]

ALLOWED_POOLING = ('avg', 'max', None)

WEIGHTS = {
  'imagenet': {
    'file_name': 'resnet38d_imagenet.h5',
    'file_hash': 'cf33dc7e57150679ecf288033c1150dec4b08eed83d291c02c9fa341596d1d9b',
    'url': 'https://github.com/lucasdavid/resnet38d-tf/releases/download/0.0.1/resnet38d_imagenet.h5',
  },
  'voc2012': {
    'file_name': 'resnet38d_voc2012.h5',
    'file_hash': 'a33cbbdca220cd35c011c1a58936e448a2b1b32833d1b4114b826496d972a749',
    'url': 'https://github.com/lucasdavid/resnet38d-tf/releases/download/0.0.1/resnet38d_voc2012.h5',
  },
}


def block(
  x: tf.Tensor,
  filters: int = 256,
  mid_filters: Optional[int] = None,
  kernel_size: S = 3,
  dilation: S = 1,
  mid_dilation: Optional[S] = None,
  strides: S = 1,
  padding: str = 'same',
  use_bias: bool = False,
  kernel_regularizer: Optional[str] = None,
  kernel_initializer: Optional[str] = 'he_normal',
  activation: str = 'relu',
  name: Optional[str] = None,
):
  cvargs = dict(kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=use_bias)

  if mid_filters is None: mid_filters = filters
  if mid_dilation is None: mid_dilation = dilation

  b2 = BatchNormalization(name=f'{name}/b0/bn')(x)
  b2 = Activation(activation, name=f'{name}/b0/ac')(b2)

  if b2.shape[-1] != filters or strides != 1:
    b1 = Conv2D(filters, kernel_size=1, strides=strides, name=f'{name}/b1/cv', **cvargs)(b2)
  else:
    b1 = x

  b2 = Conv2D(mid_filters, kernel_size=kernel_size, strides=strides, padding=padding, dilation_rate=mid_dilation, name=f'{name}/b2/cv', **cvargs)(b2)
  b2 = BatchNormalization(name=f'{name}/b2/bn')(b2)
  b2 = Activation(activation, name=f'{name}/b2/ac')(b2)
  b2 = Conv2D(filters, kernel_size=kernel_size, padding=padding, dilation_rate=dilation, name=f'{name}/b3/cv', **cvargs)(b2)

  return Add(name=f'{name}/add')([b1, b2])


def bottleneck(
  x: tf.Tensor,
  filters: int = 256,
  kernel_size: S = 3,
  dilation: S = 1,
  strides: S = 1,
  padding: str = 'same',
  use_bias: bool = False,
  kernel_regularizer: Optional[str] = None,
  kernel_initializer: str = 'he_normal',
  activation: str = 'relu',
  dropout_rate: float = 0.,
  name: Optional[str] = None,
):
  cvargs = dict(kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=use_bias)

  b2 = BatchNormalization(name=f'{name}/b0/bn')(x)
  b2 = Activation(activation, name=f'{name}/b0/ac')(b2)

  b1 = Conv2D(filters, kernel_size=1, strides=strides, name=f'{name}/b1/cv', **cvargs)(b2)

  b2 = Conv2D(filters // 4, kernel_size=1, strides=strides, name=f'{name}/b2/cv', **cvargs)(b2)
  b2 = BatchNormalization(name=f'{name}/b2/bn')(b2)
  b2 = Activation(activation, name=f'{name}/b2/ac')(b2)
  b2 = Dropout(dropout_rate, name=f'{name}/b2/dr')(b2)

  b3 = Conv2D(filters // 2, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding=padding, name=f'{name}/b3/cv', **cvargs)(b2)
  b3 = BatchNormalization(name=f'{name}/b3/bn')(b3)
  b3 = Activation(activation, name=f'{name}/b3/ac')(b3)
  b3 = Dropout(dropout_rate, name=f'{name}/b3/dr')(b3)

  b4 = Conv2D(filters, kernel_size=1, padding=padding, name=f'{name}/b4/cv', **cvargs)(b3)

  return Add(name=f'{name}/add')([b1, b4])


def ResNet38d(
  input_tensor: Optional[Union[tf.Tensor, tf.keras.Input]] = None,
  input_shape: Tuple[int, int, int] = (448, 448, 3),
  classes: int = 1000,
  include_top: int = True,
  weights: str = 'imagenet',
  dropout_rate: float = 0,
  dilation: Optional[S] = (2, 4),
  pooling: str = 'avg',
  activation: str = 'softmax',
  name: str = 'resnet38d',
):
  if pooling not in ALLOWED_POOLING:
    raise ValueError('Illegal value for the `pooling` param. '
                     f'Allowed are {ALLOWED_POOLING}')

  if isinstance(dilation, int) and dilation == 1:
    dilation = (dilation // 2, dilation)
  if not isinstance(dilation, (list, tuple)) or len(dilation) != 2:
    raise ValueError(
      f'Illegal value for the `dilation` param "{dilation}". Must be an int '
      'or a pair `(mid, end)` containing the mid and end dilation rates.')

  if input_tensor is None:
    input_tensor = tf.keras.layers.Input(shape=input_shape)
  elif not tf.keras.backend.is_keras_tensor(input_tensor):
    input_tensor = tf.keras.layers.Input(tensor=input_tensor)

  x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False, name='s0/g0/cv')(input_tensor)

  x = block(x, 128, name='s1/g0', strides=2)
  x = block(x, 128, name='s1/g1')
  x = block(x, 128, name='s1/g2')

  x = block(x, 256, name='s2/g0', strides=2)
  x = block(x, 256, name='s2/g1')
  x = block(x, 256, name='s2/g2')

  x = block(x, 512, name='s3/g0', strides=2)
  x = block(x, 512, name='s3/g1')
  x = block(x, 512, name='s3/g2')
  x = block(x, 512, name='s3/g3')
  x = block(x, 512, name='s3/g4')
  x = block(x, 512, name='s3/g5')

  x = block(x, 1024, 512, name='s4/g0', dilation=dilation[0], strides=1, mid_dilation=1)
  x = block(x, 1024, 512, name='s4/g1', dilation=dilation[0])
  x = block(x, 1024, 512, name='s4/g2', dilation=dilation[0])

  x = bottleneck(x, 2048, name='s5/b0', dilation=dilation[1], dropout_rate=dropout_rate, strides=1)
  x = bottleneck(x, 4096, name='s5/b1', dilation=dilation[1], dropout_rate=dropout_rate)
  x = BatchNormalization(name='s5/bn')(x)
  x = Activation('relu', name='s5/ac')(x)

  if include_top:
    x = Conv2D(classes, kernel_size=1, padding='same', use_bias=False, name='logits')(x)

  if pooling:
    if pooling == 'avg':
      x = GlobalAveragePooling2D(name='avg_pool')(x)
    else:
      x = GlobalMaxPooling2D(name='max_pool')(x)

  if include_top and activation:
    x = Activation(activation, name='predictions')(x)

  model = Model(get_source_inputs(input_tensor), x, name=name)

  if weights is not None:
    if weights not in WEIGHTS:
      weights_path = weights
    else:
      config = WEIGHTS[weights]

      weights_path = get_file(
        config['file_name'],
        config['url'],
        cache_subdir='models',
        file_hash=config['file_hash']
      )

    model.load_weights(weights_path)

  return model
