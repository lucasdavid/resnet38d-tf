# resnet38d-tf
Tensorflow implementation for ResNet38 dilated, with weights.

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
