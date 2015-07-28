__author__ = 'jcorrea'

# neon dependencies
from neon.backends import gen_backend
from neon.layers import FCLayer, DataLayer, CostLayer
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

def model_gen(lrate, lrule_coef, num_epochs, batch_size):
    layers = []

    layers.append(DataLayer(name = 'd0', nout=35392))

    layers.append(FCLayer(
            name = 'h0',
            nout=200,
            lrule_init={'lr_params': {'learning_rate': lrate,
                'momentum_params': {'coef': lrule_coef, 'type': 'constant'}},
                'type': 'gradient_descent_momentum'},
            weight_init=val_init.UniformValGen(low=-0.1,high=0.1),
            activation=RectLin()
        )
    )

    layers.append(FCLayer(
            name = 'output',
            nout = 2,
            lrule_init={'lr_params': {'learning_rate': lrate,
                'momentum_params': {'coef': lrule_coef, 'type': 'constant'}},
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

def hyperopt(lrate, lrule_coef, num_epochs, batch_size):

    basepath = "/Users/DOE6903584/NERSC/mantissa-new/AR/data"
    fland = os.path.join(basepath, "landmask_imgs.pkl")
    far = os.path.join(basepath, "atmosphericriver_TMQ.h5")

    dataset = AR(fland=fland, far=far)

    experiment = FitPredictErrorExperiment(model=model_gen(lrate, lrule_coef, num_epochs, batch_size), backend=gen_backend(),dataset=dataset)

    metrics = experiment.run()

    result = float(metrics['test']['MisclassPercentage_TOP_1'])

    return {'hyperopt': result}

# Write a function like this called 'main'
def main(job_id, params):

    print 'Anything printed here will end up in the output directory for job #%d' % job_id
    print params
    return hyperopt(params['lrate'], params['lrule_coef'], params['num_epochs'], params['batch_size'])

