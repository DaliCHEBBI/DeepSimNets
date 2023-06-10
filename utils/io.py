import os
import numpy as np
import utils.utils as util
#import cv2
import torchvision
import argparse
import torch
import random
import os
import yaml
from orderedattrdict.yamlutils import AttrDictYAMLLoader
#import imageio


"""
def load_image(path, res=None):
    path = str(path)

    if not os.path.isfile(path):
        raise FileNotFoundError(str(path))

    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if res is not None:
        height, width = res
        image = cv2.resize(image, (width, height))

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise NotImplementedError

    if image.ndim == 2:
        image = image[..., np.newaxis]

    if image is None:
        raise FileNotFoundError

    if image.dtype == np.uint8 or image.dtype == np.uint16:
        image = util.convert_to_float(image)

    return image


def write_gif(path, images, n_row, duration=0.1, loop=0):
    np_images = []
    for image in images:
        image = torchvision.utils.make_grid(image, n_row)
        image = util.pytorch_to_numpy(image, is_batch=False)
        image = util.convert_to_int(image)
        np_images.append(image)

    imageio.mimwrite(path, np_images, duration=duration, loop=loop)


def write_images(path, images, n_row):
    image = torchvision.utils.make_grid(images, n_row)
    image = util.pytorch_to_numpy(image, is_batch=False)
    image = util.convert_to_int(image)
    if image.ndim == 3:
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite('{}'.format(str(path)), np.squeeze(image))

"""

def load_config_train(root_dir, experiment_name, config_path=None):
    if config_path is None:
        config_path = 'configs/{}/config_default.yaml'.format(experiment_name)
    path = config_path

    config = load_config(root_dir, path)
    print("configuration file ",config)

    # set seed
    seed = config['train']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    config['feature']['channels'] = 1  # PAN
    param = config

    if param.device == 'cuda' and not torch.cuda.is_available():
        raise Exception('No GPU found, please use "cpu" as device')

    param['version_id'] = 0
    param['version'] = '{}_{}'.format(param['version_id'], param['version_name'])
    param['experiment_name'] = experiment_name
    param["device_id"]=config['device']

    return param


def load_config(root_dir, config_path):
    print('Use pytorch {}'.format(torch.__version__))
    print('Load config: {}'.format(config_path))

    config = yaml.load(open(str(config_path)), Loader=AttrDictYAMLLoader)

    config['root_dir'] = str(root_dir)

    return config
