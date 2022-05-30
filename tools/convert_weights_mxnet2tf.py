"""Convert pretrained weights in mxnet format to TensorFlow.

Useful when restoring ImageNet weights from https://github.com/itijyou/ademxapp, for example.

"""

import mxnet
import torch
from resnet38d import ResNet38d

from convert_weights_torch2tf import assign_weights, load_param_map

WEIGHTS_TORCH = './weights/resnet_38d.params'
WEIGHTS_TF = './weights/resnet38d_imagenet.h5'


def convert_mxnet_to_torch(filename):
  save_dict = mxnet.nd.load(filename)
  renamed_dict = dict()
  bn_param_mx_pt = {
    'beta': 'bias',
    'gamma': 'weight',
    'mean': 'running_mean',
    'var': 'running_var'
  }
  for k, v in save_dict.items():
    v = torch.from_numpy(v.asnumpy())
    toks = k.split('_')
    if 'conv1a' in toks[0]:
      renamed_dict['conv1a.weight'] = v
    elif 'linear1000' in toks[0]:
      pass
    elif 'branch' in toks[1]:
      pt_name = []
      if toks[0][-1] != 'a':
        pt_name.append('b' + toks[0][-3] + '_' + toks[0][-1])
      else:
        pt_name.append('b' + toks[0][-2])
      if 'res' in toks[0]:
        layer_type = 'conv'
        last_name = 'weight'
      else:  # 'bn' in toks[0]:
        layer_type = 'bn'
        last_name = bn_param_mx_pt[toks[-1]]
      pt_name.append(layer_type + '_' + toks[1])
      pt_name.append(last_name)
      torch_name = '.'.join(pt_name)
      renamed_dict[torch_name] = v
    else:
      last_name = bn_param_mx_pt[toks[-1]]
      renamed_dict['bn7.' + last_name] = v
  return renamed_dict


def main():
  rn38d = ResNet38d(input_shape=[512, 512, 3], pooling=None, include_top=False, weights=None)
  assign_weights(
    rn38d,
    weights=convert_mxnet_to_torch(WEIGHTS_TORCH),
    tf2torch=load_param_map()
  )
  rn38d.save_weights(WEIGHTS_TF)


if __name__ == '__main__':
  main()
