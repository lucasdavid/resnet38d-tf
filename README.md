# resnet38d-tf
Tensorflow implementation for ResNet38 dilated, with weights converted from the Torch implementation.

### Pretrained Weights

Weights were ported from mxnet and Torch using the conversion scripts in [tools](/tools).

  \# | Dataset | Source
---- | -------  | ------
  1 | Imagenet | [itijyou/ademxapp](https://github.com/itijyou/ademxapp)
  2 | Pascal VOC 2012 | [jiwoon-ahn/psa/res38_cls.pth](https://github.com/jiwoon-ahn/psa)

### Usage

Loading Imagenet weights:
```py
import tensorflow as tf
from resnet38d import ResNet38d

input_tensor = tf.keras.Input([512, 512, 3], name='inputs')
rn38d = ResNet38d(input_tensor=input_tensor, weights='imagenet', include_top=False)
```

Loading Pascal VOC 2012 weights:
```py
import tensorflow as tf
from resnet38d import ResNet38d

input_tensor = tf.keras.Input([512, 512, 3], name='inputs')
rn38d = ResNet38d(input_tensor=input_tensor, weights='voc2012')
```

#### Preprocessing

For both cases, data must be preprocessed with
`tf.keras.applications.imagenet_utils.preprocess_input(x, mode='torch')`.
In other words:
```py
x = load_data()

x /= 255
x -= tf.convert_to_tensor([0.485, 0.456, 0.406])
x /= tf.convert_to_tensor([0.229, 0.224, 0.225])
```
