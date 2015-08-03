__author__ = 'jcorrea'

# neon dependencies
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer, ConvLayer, PoolingLayer
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.experiments import FitPredictErrorExperiment
from neon.params import val_init
import os
from AR.model import AR
# from model import AR

import numpy as np

import logging
logging.basicConfig(level=20)
logger = logging.getLogger()

def model_gen():
    layers = []

    layers.append(DataLayer(name = 'd0',
                            is_local=True,
                            nofm=2,
                            ofmshape=[158, 224]))
      #
      # !obj:layers.ConvLayer {
      #   name: conv1,
      #   backend_type: *bt,
      #   batch_norm: *bn,
      #   lrule_init: *gdmwd,
      #   brule_init: *gdm,
      #   weight_init: !obj:params.AutoUniformValGen {
      #     relu: True,
      #     islocal: True,
      #     bias_init: 0.0,
      #   },
      #   nofm: 64,
      #   fshape: [11, 11],
      #   stride: 4,
      #   pad: 3,
      #   activation: !obj:transforms.RectLin {},
      # },

# type: gradient_descent_momentum_weight_decay,
#     backend_type: *bt,
#     lr_params: {
#       learning_rate: 0.0001,
#       weight_decay: .0005,
#       schedule: {
#               type: step,
#               ratio: 0.1,
#               step_epochs: 10,
#       },
#       momentum_params: {
#         type: constant,
#         initial_coef: 0.90,
#       },
#     },
#   },

    layers.append(ConvLayer(
        name = 'conv1',
        backend_type = {'batch_norm': True},
        lrule_init={'lr_params': {'learning_rate': 0.0001,
                                  'weight_decay': .0005,
                                  'schedule': {'type': 'step',
                                               'ratio': 0.1,
                                               'step_epochs': 10},
        'momentum_params': {'initial_coef': 0.9, 'type': 'constant'}},
        'type': 'gradient_descent_momentum_weight_decay'}

    ))

    # xx_size = 158
    # yy_size = 224
    # layers.append(DataLayer(name = 'd0', nout=35392))


      # &datalayer !obj:layers.ImageDataLayer {
      #   name: d0,
      #   is_local: True,
      #   nofm: 3,
      #   ofmshape: [*cis, *cis],
      # },

    # lrule: &gdmwd {
    # type: gradient_descent_momentum_weight_decay,
    # backend_type: *bt,
    # lr_params: {
    #   learning_rate: 0.0001,
    #   weight_decay: .0005,
    #   schedule: {
    #           type: step,
    #           ratio: 0.1,
    #           step_epochs: 10,
    #   },
    #   momentum_params: {
    #     type: constant,
    #     initial_coef: 0.90,
    #   },
    # },

    # layers.append(ConvLayer(
    #     name = 'layer1',
    #     nofm = 16,
    #     fshape = [5,5],
    #     weight_init = val_init.UniformValGen(low=-0.1,high=0.1),
    #     lrule_init={'lr_params': {'learning_rate': 0.01,
    #             'momentum_params': {'coef': 0.9, 'type': 'constant'}},
    #             'type': 'gradient_descent_momentum'}
    # ))

    # layers.append(ConvLayer(
    #     name = 'layer1',
    #     nofm = 16,
    #     fshape = [5,5],
    #     weight_init = val_init.UniformValGen(low=-0.1,high=0.1),
    #     lrule_init={'lr_params': {'learning_rate': 0.01,
    #             'momentum_params': {'coef': 0.9, 'type': 'constant'}},
    #             'type': 'gradient_descent_momentum'}
    # ))

# !obj:layers.ConvLayer {
#         name: conv1,
#         backend_type: *bt,
#         batch_norm: *bn,
#         lrule_init: *gdmwd,
#         brule_init: *gdm,
#         weight_init: !obj:params.AutoUniformValGen {
#           relu: True,
#           islocal: True,
#           bias_init: 0.0,
#         },
#         nofm: 64,
#         fshape: [11, 11],
#         stride: 4,
#         pad: 3,
#         activation: !obj:transforms.RectLin {},
#       },

    layers.append(PoolingLayer(
        name='layer2',
        op = 'max',
        fshape = [2,2],
        stride = 2
    ))

    layers.append(ConvLayer(
        name = 'layer3',
        nofm = 32,
        fshape = [5,5],
        weight_init = val_init.UniformValGen(low=-0.1,high=0.1),
        lrule_init={'lr_params': {'learning_rate': 0.01,
                'momentum_params': {'coef': 0.9, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'}
    ))

    layers.append(PoolingLayer(
        name='layer4',
        op = 'max',
        fshape = [2,2],
        stride = 2
    ))

    layers.append(FCLayer(
            name = 'layer5',
            nout=500,
            lrule_init={'lr_params': {'learning_rate': 0.01,
                'momentum_params': {'coef': 0.9, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'},
            weight_init=val_init.UniformValGen(low=-0.1,high=0.1),
            activation=RectLin()
        )
    )

    layers.append(FCLayer(
            name = 'output',
            nout = 2,
            lrule_init={'lr_params': {'learning_rate': 0.01,
                'momentum_params': {'coef': 0.9, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'},
            weight_init=val_init.UniformValGen(low=-0.1,high=0.1),
            activation = Logistic()
        )
    )

    layers.append(CostLayer(
            name = 'cost',
            ref_layer = layers[0],
            cost = CrossEntropy()
        )
    )
    model = MLP(num_epochs=10, batch_size=200, layers=layers)

    return model


basepath = "/project/projectdirs/nervana/berghain/data"
# basepath = "/Users/DOE6903584/NERSC/mantissa-new/AR/data"
repo_path = os.path.join(basepath, "/results")
fland = os.path.join(basepath, "landmask_imgs.pkl")
far = os.path.join(basepath, "atmosphericriver_TMQ.h5")

dataset = AR(fland=fland, far=far)

experiment = FitPredictErrorExperiment(model=model_gen(), backend=gen_backend(),dataset=dataset,repo_path=repo_path)

experiment.run()