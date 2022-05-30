"""Convert pretrained weights in Torch format to TensorFlow.

Useful when restoring ImageNet weights from https://github.com/jiwoon-ahn/psa, for example.

"""

import re

import torch

from resnet38d import ResNet38d


WEIGHTS_TORCH = './weights/014net_er.pth'
WEIGHTS_TF = './weights/resnet38d_occse_voc2012.h5'

PARAM_NAMES_FILE = './resnet38d.param_names'


def main():
  rn38d = ResNet38d(input_shape=[512, 512, 3], weights=None)
  assign_weights(
    rn38d,
    weights=torch.load(WEIGHTS_TORCH, map_location='cpu'),
    tf2torch=load_param_map()
  )
  rn38d.save_weights(WEIGHTS_TF)


def load_param_map():
  with open(PARAM_NAMES_FILE) as f:
    params = f.readlines()

  params = [re.split('\s+', p.strip()) for p in params]
  return dict(params)


def assign_weights(model, weights, tf2torch):
  for w in model.weights:
    print(w.name, end=' ')
    if w.name not in tf2torch:
      print(f' {w.name} not found in tf2torch')
      continue
    torch_name = tf2torch[w.name]
    if torch_name not in weights:
      print(f' {torch_name} not found in weights')
      continue
    v = weights[torch_name].numpy()
    if 'kernel' in w.name and w.shape.rank == 4:
      v = v.transpose((2, 3, 1, 0))
    w.assign(v)

    print('s:', v.shape)


if __name__ == '__main__':
  main()
