__author__ = 'jcorrea'

import numpy as np
import h5py
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
import pickle
import logging
import os

from neon.datasets.dataset import Dataset

logger = logging.getLogger(__name__)

class AR(Dataset):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        # kwargs:   fland   -   path/to/mask
        #           far     -   path/to/dataset

        # self.data_train = self.data['data_train']
        # self.data_test = self.data['data_test']
        # self.labels_train = self.data['labels_train']
        # self.labels_test = self.data['labels_test']
        self.initialize()

    def initialize(self):
        dland = np.asarray(pickle.load(open(self.fland, 'r'))['mask'])
        darnar = h5py.File(self.far, 'r')

        dar = np.asarray(darnar['AR'])
        dnar = np.asarray(darnar['Non_AR'])

        # Land-mask, AR, nAR
        # f = plt.subplots()
        # plt.subplot(3, 3, 1)
        # plt.imshow(dland[0])
        # plt.axis('off')

        # plt.subplot(3, 3, 2)
        # plt.imshow(dar[0][0])
        # plt.axis('off')
        #
        # plt.subplot(3, 3, 3)
        # plt.imshow(dnar[0][0])
        # plt.axis('off')

        # TMQ thresholding
        tmq_thr = 20
        dar_i = np.multiply(dar, dland).clip(tmq_thr)
        dnar_i = np.multiply(dnar, dland).clip(tmq_thr)

        # tmq_thr = 20
        # dar_i = np.multiply(dar, dland)
        # dnar_i = np.multiply(dnar, dland)
        #
        # dar_i = dar_i[dnar_i>=tmq_thr]
        # dnar_i = dnar_i[dnar_i>=tmq_thr]

        # Thr LM.*AR/LM.*nAR
        # f = plt.subplots()

        # Thresholded AR
        # plt.subplot(2, 2, 1)
        # plt.imshow(dar_i[0][0])
        # plt.axis('off')

        # Thresholded nAR
        # plt.subplot(2, 2, 2)
        # plt.imshow(dnar_i[0][0])
        # plt.axis('off')

        # TR/TE sizes for AR/nAR

        tr_size_ar = 2000
        tr_size_nar = 2000
        #
        te_size_ar = 468
        te_size_nar = 1077

        # tr_size_ar = 100
        # tr_size_nar = 100
        #
        # te_size_ar = 10
        # te_size_nar = 10

        l_ar = np.ones(tr_size_ar + te_size_ar)
        l_nar = np.zeros(tr_size_nar + te_size_nar)

        dar = dar_i
        dnar = dnar_i

        s = range(len(dar))
        d = range(len(dnar))
        np.random.shuffle(s)
        np.random.shuffle(d)

        tr_ar = np.squeeze(dar[s][:][:][:tr_size_ar])
        tr_nar = np.squeeze(dnar[d][:][:][:tr_size_nar])

        te_ar = np.squeeze(dar[s][:][:][-te_size_ar:])
        te_nar = np.squeeze(dnar[d][:][:][-te_size_nar:])

        # Some data manipulation
        Ftr_ar = np.asarray([normalize(tr_ar[i]).flatten() for i in range(len(tr_ar))])
        Ftr_nar = np.asarray([normalize(tr_nar[i]).flatten() for i in range(len(tr_nar))])
        Fte_ar = np.asarray([normalize(te_ar[i]).flatten() for i in range(len(te_ar))])
        Fte_nar = np.asarray([normalize(te_nar[i]).flatten() for i in range(len(te_nar))])

        d_tr = np.vstack(([Ftr_ar, Ftr_nar]))
        d_te = np.vstack(([Fte_ar, Fte_nar]))

        # This labels are correct
        l_tr = np.vstack(([[1,0]] * tr_size_ar,
                        [[0,1]] * tr_size_nar))

        l_te = np.vstack(([[1,0]] * te_size_ar,
                        [[0,1]] * te_size_nar))

        xx_size = 158
        yy_size = 224

        self.data_train = d_tr
        self.data_test = d_te
        self.labels_train = l_tr
        self.labels_test = l_te

    def load(self, backend=None, experiment=None):
        # Dataset.inputs

#         self.backend = None
        self.inputs = {'train': self.data_train,
                       'test': self.data_test,
                       'validation': self.data_test[:5]}

        self.targets = {'train': self.labels_train,
                        'test': self.labels_test,
                        'validation': self.labels_test[:5]}

        self.format()


