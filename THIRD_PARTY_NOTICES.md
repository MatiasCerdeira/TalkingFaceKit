# Third-party notices

## SMIRK preparation and encoder

`src/talkingfacekit/integrations/smirk.py` adapts the MediaPipe configuration and face crop from
`utils/mediapipe_utils.py` and `demo_video.py`; `_smirk_model.py` adapts the network architecture in
`src/smirk_encoder.py`, all from
https://github.com/georgeretsi/smirk/tree/c7de404c4389f073906a6db1adabf62efcea3f35.
The MIT license is included in `src/talkingfacekit/integrations/SMIRK_LICENSE.txt`.
Backbone initialization disables pretrained ImageNet acquisition and requires a complete SMIRK
state dictionary. No SMIRK checkpoint, FLAME asset or renderer is distributed with the package.

## FLAME2020 geometry and model assets

The NumPy decoder in `integrations/flame.py`, `_flame_model.py` and `_flame_assets.py` implements
the FLAME blend-shape and articulated linear blend skinning model. FLAME reference:
Li et al., "Learning a model of facial shape and expression from 4D scans", SIGGRAPH Asia 2017,
https://doi.org/10.1145/3130800.3130813.

Geometry is numerically compared with the external SMIRK FLAME decoder at the revision above;
the Max Planck FLAME/LBS source files are not included in this package. The official FLAME2020
model and SMIRK eyelid deformation files are explicit local assets, not distributed here. Their
own terms apply independently from this library's source code:
https://flame.is.tue.mpg.de/modellicense.html.

## SyncNet-v2

`src/talkingfacekit/integrations/_syncnet_model.py` adapts the network architecture in
https://github.com/joonson/syncnet_python/blob/master/SyncNetModel.py.
Face cropping and feature-distance scoring in the SyncNet adapter follow the same repository.
Model weights are not distributed with TalkingFaceKit. Official checkpoint:
https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model.

Copyright (c) 2016-present Joon Son Chung.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
