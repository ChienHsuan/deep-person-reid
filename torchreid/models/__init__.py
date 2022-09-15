import torch

from .resnet import resnet50, resnet101
from .resnet_ibn import resnet_ibn50a, resnet_ibn101a
from .osnet import osnet_x1_0, osnet_x0_5
from .osnet_ain import osnet_ain_x1_0, osnet_ain_x0_5


__model_factory = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet_ibn50a': resnet_ibn50a,
    'resnet_ibn101a': resnet_ibn101a,
    'osnet_x1_0': osnet_x1_0,
    'osnet_x0_5': osnet_x0_5,
    'osnet_ain_x1_0': osnet_ain_x1_0,
    'osnet_ain_x0_5': osnet_ain_x0_5,
}

def show_avai_models():
    """Displays available models.

    Examples::
        >>> from torchreid import models
        >>> models.show_avai_models()
    """
    print(list(__model_factory.keys()))


def build_model(
    name, num_classes, feature_dim=512, pretrained=True
):
    """A function wrapper for building a model.
    """
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    
    return __model_factory[name](
        num_classes=num_classes,
        feature_dim=feature_dim,
        pretrained=pretrained
    )
