__author__ = 'jcorrea'

# neon dependencies
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer, ConvLayer, PoolingLayer
from neon.models import MLP
from neon.transforms import RectLin, Logistic, CrossEntropy
from neon.experiments import FitPredictErrorExperiment
from neon.params import val_init
import os
from model import AR

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

    # xx_size = 158
    # yy_size = 224
    # layers.append(DataLayer(name = 'd0', nout=35392))


      # &datalayer !obj:layers.ImageDataLayer {
      #   name: d0,
      #   is_local: True,
      #   nofm: 3,
      #   ofmshape: [*cis, *cis],
      # },

    layers.append(ConvLayer(
        name = 'layer1',
        nofm = 16,
        fshape = [5,5],
        weight_init = val_init.UniformValGen(low=-0.1,high=0.1),
        lrule_init={'lr_params': {'learning_rate': 0.01,
                'momentum_params': {'coef': 0.9, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'}
    ))


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
    model = MLP(num_epochs=2, batch_size=200, layers=layers)

    return model


def hyperopt(lrate, lrule_coef, num_epochs, batch_size):

    basepath = "/Users/DOE6903584/NERSC/mantissa-new/AR/data"
    fland = os.path.join(basepath, "landmask_imgs.pkl")
    far = os.path.join(basepath, "atmosphericriver_TMQ.h5")

    dataset = AR(fland=fland, far=far)
    # metrics = {'train':[LogLossMean(), MisclassRate(), MSE()], 'validation':[LogLossMean(), MisclassRate(), MSE()], 'test':[]}
    experiment = FitPredictErrorExperiment(model=model_gen(lrate, lrule_coef, num_epochs, batch_size), backend=gen_backend(),dataset=dataset)

    metrics = experiment.run()

    result = float(metrics['test']['MisclassPercentage_TOP_1'])

    # # Experiment result is dict: result[metric_set][metric_name]
    # result = experiment.run()
    # for setname in result.keys():
    #     for metricname in result[setname].keys():
    #         print '%s_%s: %f' % (setname, metricname, result[setname][metricname])
    # loss = result['validation']['MSE']
    # return loss

    return {'hyperopt': result}

# Write a function like this called 'main'
def main(job_id, params):

    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return hyperopt(params['lrate'], params['lrule_coef'], params['num_epochs'], params['batch_size'])