import argparse

import models.sr.ersgan.Train as ersgan
import models.sr.srgan.Train as srgan
import models.generative.context_encoder.Train as ce
import models.cnn.resnet.Train as resnet

models = ["ersgan", "srgan", "context_encoder", "resnet"]

