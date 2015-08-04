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

import numpy as np


import logging
logging.basicConfig(level=20)
logger = logging.getLogger()

# def model_gen(nofm_layer1, fshape_layer1, learning_rate_layer1, coef_layer1,
#               fshape_layer2, stride_layer2, nofm_layer3, fshape_layer3,
#               learning_rate_layer3, coef_layer3, fshape_layer4, stride_layer4,
#               learning_rate_layer5, coef_layer5, learning_rate_output, coef_output,
#               num_epochs, batch_size):

def model_gen(params):

    nofm_layer1=params['nofm_layer1']
    fshape_layer1=params['fshape_layer1']
    learning_rate_layer1=params['learning_rate_layer1']
    coef_layer1=params['coef_layer1']
    fshape_layer2=params['fshape_layer2']
    stride_layer2=params['stride_layer2']
    nofm_layer3=params['nofm_layer3']
    fshape_layer3=params['fshape_layer3']
    learning_rate_layer3=params['learning_rate_layer3']
    coef_layer3=params['coef_layer3']
    fshape_layer4=params['fshape_layer4']
    stride_layer4=params['stride_layer4']
    learning_rate_layer5=params['learning_rate_layer5']
    coef_layer5=params['coef_layer5']
    learning_rate_output=params['learning_rate_output']
    coef_output=params['coef_output']
    num_epochs=params['num_epochs']
    batch_size=params['batch_size']

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
        nofm = nofm_layer1,
        fshape = [fshape_layer1,fshape_layer1],
        weight_init = val_init.UniformValGen(low=-0.1,high=0.1),
        lrule_init={'lr_params': {'learning_rate': learning_rate_layer1,
                'momentum_params': {'coef': coef_layer1, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'}
    ))


    layers.append(PoolingLayer(
        name='layer2',
        op = 'max',
        fshape = [fshape_layer2,fshape_layer2],
        stride = stride_layer2
    ))

    layers.append(ConvLayer(
        name = 'layer3',
        nofm = nofm_layer3,
        fshape = [fshape_layer3,fshape_layer3],
        weight_init = val_init.UniformValGen(low=-0.1,high=0.1),
        lrule_init={'lr_params': {'learning_rate': learning_rate_layer3,
                'momentum_params': {'coef': coef_layer3, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'}
    ))

    layers.append(PoolingLayer(
        name='layer4',
        op = 'max',
        fshape = [fshape_layer4,fshape_layer4],
        stride = stride_layer4
    ))


    nout_layer5 = 500
    layers.append(FCLayer(
            name = 'layer5',
            nout=nout_layer5,
            lrule_init={'lr_params': {'learning_rate': learning_rate_layer5,
                'momentum_params': {'coef': coef_layer5, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'},
            weight_init=val_init.UniformValGen(low=-0.1,high=0.1),
            activation=RectLin()
        )
    )

    layers.append(FCLayer(
            name = 'output',
            nout = 2,
            lrule_init={'lr_params': {'learning_rate': learning_rate_output,
                'momentum_params': {'coef': coef_output, 'type': 'constant'}},
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
    model = MLP(num_epochs=num_epochs, batch_size=batch_size, layers=layers)

    return model

#
# def hyperopt(nofm_layer1, fshape_layer1, learning_rate_layer1, coef_layer1,
#            fshape_layer2, stride_layer2, nofm_layer3, fshape_layer3,
#            learning_rate_layer3, coef_layer3, fshape_layer4, stride_layer4,
#             learning_rate_layer5, coef_layer5, learning_rate_output, coef_output,
#            num_epochs, batch_size):

def hyperopt(params):

    # nofm_layer1=params['nofm_layer1']
    # fshape_layer1=params['fshape_layer1']
    # learning_rate_layer1=params['learning_rate_layer1']
    # coef_layer1=params['coef_layer1']
    # fshape_layer2=params['fshape_layer2']
    # stride_layer2=params['stride_layer2']
    # nofm_layer3=params['nofm_layer3']
    # fshape_layer3=params['fshape_layer3']
    # learning_rate_layer3=params['learning_rate_layer3']
    # coef_layer3=params['coef_layer3']
    # fshape_layer4=params['fshape_layer4']
    # stride_layer4=params['stride_layer4']
    # learning_rate_layer5=params['learning_rate_layer5']
    # coef_layer5=params['coef_layer5']
    # learning_rate_output=params['learning_rate_output']
    # coef_output=params['coef_output']
    # num_epochs=params['num_epochs']
    # batch_size=params['batch_size']

    # basepath = "/Users/DOE6903584/NERSC/mantissa-new/AR/data"
    # repo_path = os.path.join(basepath, "/results/")
    # fland = os.path.join(basepath, "landmask_imgs.pkl")
    # far = os.path.join(basepath, "atmosphericriver_TMQ.h5")

    basepath = "/project/projectdirs/nervana/berghain/data"
    # basepath = "/Users/DOE6903584/NERSC/mantissa-new/AR/data"
    repo_path = os.path.join(basepath, "/results/")
    fland = os.path.join(basepath, "landmask_imgs.pkl")
    far = os.path.join(basepath, "atmosphericriver_TMQ.h5")

    # dataset = AR(fland=fland, far=far, save_dir=repo_path, repo_path=repo_path)
    # dataset = AR(fland=fland, far=far, save_dir=repo_path)
    dataset = AR(fland=fland, far=far)

    # metrics = {'train':[LogLossMean(), MisclassRate(), MSE()], 'validation':[LogLossMean(), MisclassRate(), MSE()], 'test':[]}
    # experiment = FitPredictErrorExperiment(
    #     model=model_gen(nofm_layer1, fshape_layer1, learning_rate_layer1, coef_layer1,
    #                                                        fshape_layer2, stride_layer2, nofm_layer3, fshape_layer3,
    #                                                        learning_rate_layer3, coef_layer3, fshape_layer4, stride_layer4,
    #                                                         learning_rate_layer5, coef_layer5, learning_rate_output, coef_output,
    #                                                        num_epochs, batch_size),
    #     backend=gen_backend(),
    #     dataset=dataset)

    experiment = FitPredictErrorExperiment(model=model_gen(params),backend=gen_backend(),dataset=dataset)

    metrics = experiment.run()

    result = float(metrics['test']['MisclassPercentage_TOP_1'] / 100)


    # # Experiment result is dict: result[metric_set][metric_name]
    # result = experiment.run()
    # for setname in result.keys():
    #     for metricname in result[setname].keys():
    #         print '%s_%s: %f' % (setname, metricname, result[setname][metricname])
    # loss = result['validation']['MSE']
    # return loss

    return {'hyperopt': result}

# Write a function like this called 'main'
# def main(job_id, params):
#
#     print 'Anything printed here will end up in the output directory for job #%d' % job_id
#
#     print params
#
#     return hyperopt(params['nofm_layer1'], params['fshape_layer1'], params['learning_rate_layer1'], params['coef_layer1'],
#                 params['fshape_layer2'], params['stride_layer2'], params['nofm_layer3'], params['fshape_layer3'],
#                 params['learning_rate_layer3'], params['coef_layer3'], params['fshape_layer4'], params['stride_layer4'],
#                 params['learning_rate_layer5'], params['coef_layer5'], params['learning_rate_output'], params['coef_output'],
#                 params['num_epochs'], params['batch_size'])

def main(job_id, params):

    print 'Anything printed here will end up in the output directory for job #%d' % job_id

    print params

    return hyperopt(params)