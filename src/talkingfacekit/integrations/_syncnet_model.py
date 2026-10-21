"""SyncNet-v2 network adapted from Joon Son Chung's MIT-licensed implementation.

Source: https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py
Copyright (c) 2016-present Joon Son Chung. See THIRD_PARTY_NOTICES.md.
PyTorch stays behind this optional boundary; module/state names match the original checkpoint.
"""

from typing import Any, cast


def build_syncnet_model(torch_module: object) -> object:
    """Build the upstream audio and visual towers with 1024-dimensional embeddings."""
    nn = cast(Any, torch_module).nn
    model = nn.Module()
    model.netcnnaud = nn.Sequential(
        nn.Conv2d(1, 64, (3, 3), padding=(1, 1)),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((1, 1)),
        nn.Conv2d(64, 192, (3, 3), padding=(1, 1)),
        nn.BatchNorm2d(192),
        nn.ReLU(),
        nn.MaxPool2d((3, 3), stride=(1, 2)),
        nn.Conv2d(192, 384, (3, 3), padding=(1, 1)),
        nn.BatchNorm2d(384),
        nn.ReLU(),
        nn.Conv2d(384, 256, (3, 3), padding=(1, 1)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, (3, 3), padding=(1, 1)),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d((3, 3), stride=(2, 2)),
        nn.Conv2d(256, 512, (5, 4)),
        nn.BatchNorm2d(512),
        nn.ReLU(),
    )
    model.netfcaud = nn.Sequential(
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1024),
    )
    model.netfclip = nn.Sequential(
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 1024),
    )
    model.netcnnlip = nn.Sequential(
        nn.Conv3d(3, 96, (5, 7, 7), stride=(1, 2, 2)),
        nn.BatchNorm3d(96),
        nn.ReLU(),
        nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
        nn.Conv3d(96, 256, (1, 5, 5), stride=(1, 2, 2), padding=(0, 1, 1)),
        nn.BatchNorm3d(256),
        nn.ReLU(),
        nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        nn.Conv3d(256, 256, (1, 3, 3), padding=(0, 1, 1)),
        nn.BatchNorm3d(256),
        nn.ReLU(),
        nn.Conv3d(256, 256, (1, 3, 3), padding=(0, 1, 1)),
        nn.BatchNorm3d(256),
        nn.ReLU(),
        nn.Conv3d(256, 256, (1, 3, 3), padding=(0, 1, 1)),
        nn.BatchNorm3d(256),
        nn.ReLU(),
        nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
        nn.Conv3d(256, 512, (1, 6, 6)),
        nn.BatchNorm3d(512),
        nn.ReLU(),
    )
    return model
