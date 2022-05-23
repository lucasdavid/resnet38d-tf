import tensorflow as tf
from keras import Model
from keras.layers import (
  Activation, BatchNormalization, Conv2D, Dropout,
  GlobalAveragePooling2D, GlobalMaxPooling2D, Add
)

ALLOWED_POOLING = ('avg', 'max', None)

WEIGHTS = {
  'imagenet': {
    'classes': 1000,
    'filename': 'resnet38d_imagenet.h5',
  },
  'voc2012': {
    'classes': 20,
    'filename': 'resnet38d_voc2012.h5',
  },
}


def block(
  x,
  filters=256,
  mid_filters=None,
  kernel_size=3,
  dilation=1,
  mid_dilation=None,
  strides=1,
  padding='same',
  use_bias=False,
  kernel_regularizer=None,
  kernel_initializer='he_normal',
  activation: str = 'relu',
  name: str = None,
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
  x,
  filters=256,
  kernel_size=3,
  dilation=1,
  strides=1,
  padding='same',
  use_bias=False,
  kernel_regularizer=None,
  kernel_initializer='he_normal',
  activation: str = 'relu',
  dropout_rate: float = 0.,
  name: str = None,
):
  cvargs = dict(kernel_regularizer=kernel_regularizer, kernel_initializer=kernel_initializer, use_bias=use_bias)

  b2 = BatchNormalization(name=f'{name}/b0/bn')(x)
  b2 = Activation(activation, name=f'{name}/b0/ac')(b2)

  b1 = Conv2D(filters, kernel_size=1, strides=strides, name=f'{name}/b1/cv', **cvargs)(b2)

  b2 = Conv2D(filters // 4, kernel_size=1, strides=strides, name=f'{name}/b2/cv', **cvargs)(b2)
  b2 = BatchNormalization(name=f'{name}/b2/bn')(b2)
  b2 = Activation(activation, name=f'{name}/b2/ac')(b2)
  b2 = Dropout(dropout_rate, name=f'{name}/b2/drop')(b2)

  b3 = Conv2D(filters // 2, kernel_size=kernel_size, strides=strides, dilation_rate=dilation, padding=padding, name=f'{name}/b3/cv', **cvargs)(b2)
  b3 = BatchNormalization(name=f'{name}/b3/bn')(b3)
  b3 = Activation(activation, name=f'{name}/b3/ac')(b3)
  b3 = Dropout(dropout_rate, name=f'{name}/b3/drop')(b3)

  b4 = Conv2D(filters, kernel_size=1, padding=padding, name=f'{name}/b4/cv', **cvargs)(b3)

  return Add(name=f'{name}/add')([b1, b4])


def ResNet38d(
  input_tensor=None,
  input_shape=None,
  classes=20,
  include_top=True,
  weights: str = "imagenet",
  dropout_rate: float = 0,
  dilated: bool = False,
  dilation: int = (2, 4),
  name: str = 'resnet38d',
  pooling: str = 'avg',
):
  if pooling not in ALLOWED_POOLING:
    raise ValueError('Illegal value for the `pooling` param. '
                     f'Allowed are {ALLOWED_POOLING}')

  if dilated:
    dilation = (2, 4)
  if isinstance(dilation, int) and dilation != 1:
    dilation = (dilation // 2, dilation)
  if isinstance(dilation, (list, tuple)):
    if len(dilation) != 2:
      raise ValueError(
        'Illegal value for the `dilation` param. Must be an int or a pair '
        '`(mid, end)` containing the mid and end dilation rates.')

  if input_tensor is None:
    input_tensor = tf.keras.layers.Input(shape=input_shape)
  elif not tf.keras.backend.is_keras_tensor(input_tensor):
    input_tensor = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)

  x = Conv2D(filters=64, kernel_size=3, padding='same', use_bias=False, name='s0/block0/cv')(input_tensor)

  x = block(x, 128, name='s1/block0', strides=2)
  x = block(x, 128, name='s1/block1')
  x = block(x, 128, name='s1/block2')

  x = block(x, 256, name='s2/block0', strides=2)
  x = block(x, 256, name='s2/block1')
  x = block(x, 256, name='s2/block2')

  x = block(x, 512, name='s3/block0', strides=2)
  x = block(x, 512, name='s3/block1')
  x = block(x, 512, name='s3/block2')
  x = block(x, 512, name='s3/block3')
  x = block(x, 512, name='s3/block4')
  x = block(x, 512, name='s3/block5')

  x = block(x, 1024, 512, name='s4/block0', dilation=dilation[0], strides=1, mid_dilation=1)
  x = block(x, 1024, 512, name='s4/block1', dilation=dilation[0])
  x = block(x, 1024, 512, name='s4/block2', dilation=dilation[0])

  x = bottleneck(x, 2048, name='s5/bottleneck0', dilation=dilation[1], dropout_rate=dropout_rate, strides=1)
  x = bottleneck(x, 4096, name='s5/bottleneck1', dilation=dilation[1], dropout_rate=dropout_rate)
  x = BatchNormalization(name='s5/bn')(x)
  x = Activation('relu', name='s5/ac')(x)

  if include_top:
    x = Conv2D(classes, kernel_size=1, padding='same', use_bias=False, name='logits')(x)

  if pooling:
    if pooling == 'avg':
      x = GlobalAveragePooling2D(name='avg_pool')(x)
    else:
      x = GlobalMaxPooling2D(name='max_pool')(x)

  model = Model(input_tensor, x, name=name)

  if weights is not None:
    if weights not in WEIGHTS:
      filename = weights
    else:
      config = WEIGHTS[weights]
      filename = config['filename']

    model.load_weights(filename)

  return model
