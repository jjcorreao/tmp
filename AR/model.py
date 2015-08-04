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

        self.dar = dar
        self.dnar = dnar

        # Correct!
        # TMQ thresholding
        tmq_thr = 20
        dar_i = np.multiply(dar, dland)
        dnar_i = np.multiply(dnar, dland)

        # Correct!
        dar_idx = dar_i <= tmq_thr
        dnar_idx = dnar_i <= tmq_thr

        # Correct!
        dar_i[dar_idx] = 0
        dnar_i[dnar_idx] = 0

        # Correct!
        self.dar_i = dar_i
        self.dnar_i = dnar_i

        # TR/TE sizes for AR/nAR

        tr_size_ar = 2000
        tr_size_nar = 2000

        te_size_ar = 468
        te_size_nar = 1077

        # tr_size_ar = 500
        # tr_size_nar = 500
        #
        # te_size_ar = 500
        # te_size_nar = 500

        # l_ar = np.ones(tr_size_ar + te_size_ar)
        # l_nar = np.zeros(tr_size_nar + te_size_nar)

        # Correct!
        dar_i = dar_i.squeeze()
        dnar_i = dnar_i.squeeze()

        # s = range(len(dar))
        # d = range(len(dnar))
        # np.random.shuffle(s)
        # np.random.shuffle(d)

        # tr_ar = np.squeeze(dar[s][:][:][:tr_size_ar])
        # tr_nar = np.squeeze(dnar[d][:][:][:tr_size_nar])
        #
        # te_ar = np.squeeze(dar[s][:][:][-te_size_ar:])
        # te_nar = np.squeeze(dnar[d][:][:][-te_size_nar:])

        # tr_ar = dar_i[:][:][:tr_size_ar]
        # tr_nar = dnar_i[:][:][:tr_size_nar]

        # Correct
        tr_ar = dar_i[:tr_size_ar]
        tr_nar = dnar_i[:tr_size_nar]

        te_ar = dar_i[-te_size_ar:]
        te_nar = dnar_i[-te_size_nar:]

        # Some data manipulation
        # Ftr_ar = np.asarray([normalize(tr_ar[i]).flatten() for i in range(len(tr_ar))])
        # Ftr_nar = np.asarray([normalize(tr_nar[i]).flatten() for i in range(len(tr_nar))])
        # Fte_ar = np.asarray([normalize(te_ar[i]).flatten() for i in range(len(te_ar))])
        # Fte_nar = np.asarray([normalize(te_nar[i]).flatten() for i in range(len(te_nar))])

        # Original data
        # dar_tr = dar.squeeze()[:][:][:tr_size_ar]
        # dnar_tr = dnar.squeeze()[:][:][:tr_size_nar]
        # dar_te = dar.squeeze()[:][:][-te_size_ar:]
        # dnar_te = dnar.squeeze()[:][:][-te_size_nar:]

        # Correct!
        dar_tr = dar.squeeze()[:tr_size_ar]
        dnar_tr = dnar.squeeze()[:tr_size_nar]
        dar_te = dar.squeeze()[-te_size_ar:]
        dnar_te = dnar.squeeze()[-te_size_nar:]

        # Correct!
        Ftr_ar = np.asarray([normalize(tr_ar[i]).flatten() for i in range(len(tr_ar))])
        Ftr_arI = np.asarray([normalize(dar_tr[i]).flatten() for i in range(len(dar_tr))])

        Ftr_nar = np.asarray([normalize(tr_nar[i]).flatten() for i in range(len(tr_nar))])
        Ftr_narI = np.asarray([normalize(dnar_tr[i]).flatten() for i in range(len(dnar_tr))])

        Fte_ar = np.asarray([normalize(te_ar[i]).flatten() for i in range(len(te_ar))])
        Fte_arI = np.asarray([normalize(dar_te[i]).flatten() for i in range(len(dar_te))])

        Fte_nar = np.asarray([normalize(te_nar[i]).flatten() for i in range(len(te_nar))])
        Fte_narI = np.asarray([normalize(dnar_te[i]).flatten() for i in range(len(dnar_te))])

        # # Correct!
        # Ftr_ar = np.asarray([tr_ar[i].flatten() for i in range(len(tr_ar))])
        # Ftr_arI = np.asarray([dar_tr[i].flatten() for i in range(len(dar_tr))])
        #
        # Ftr_nar = np.asarray([tr_nar[i].flatten() for i in range(len(tr_nar))])
        # Ftr_narI = np.asarray([dnar_tr[i].flatten() for i in range(len(dnar_tr))])
        #
        # Fte_ar = np.asarray([te_ar[i].flatten() for i in range(len(te_ar))])
        # Fte_arI = np.asarray([dar_te[i].flatten() for i in range(len(dar_te))])
        #
        # Fte_nar = np.asarray([te_nar[i].flatten() for i in range(len(te_nar))])
        # Fte_narI = np.asarray([dnar_te[i].flatten() for i in range(len(dnar_te))])



        # ja=[oe[i].flatten() for i in range(len(oe))]

        # d_tr = np.vstack(([Ftr_ar, Ftr_nar]))
        # d_te = np.vstack(([Fte_ar, Fte_nar]))

        # Correct!
        oe_ar=np.hstack(([Ftr_ar, Ftr_arI]))
        oe_nar=np.hstack(([Ftr_nar, Ftr_narI]))
        oe_ar_te=np.hstack(([Fte_ar, Fte_arI]))
        oe_nar_te=np.hstack(([Fte_nar, Fte_narI]))

        # d_tr = np.vstack(([Ftr_ar, Ftr_nar]))
        # d_te = np.vstack(([Fte_ar, Fte_nar]))

        # Correct!
        d_tr = np.vstack(([oe_ar, oe_nar]))
        d_te = np.vstack(([oe_ar_te, oe_nar_te]))

        # This labels are correct
        l_tr = np.vstack(([[1,0]] * tr_size_ar,
                        [[0,1]] * tr_size_nar))

        l_te = np.vstack(([[1,0]] * te_size_ar,
                        [[0,1]] * te_size_nar))

        xx_size = 158
        yy_size = 224

        # self.s = range(len(d_tr))
        # self.d = range(len(d_te))
        # np.random.shuffle(self.s)
        # np.random.shuffle(self.d)
        #
        # self.data_train = d_tr[self.s]
        # self.data_test = d_te[self.d]
        # self.labels_train = l_tr[self.s]
        # self.labels_test = l_te[self.d]

        self.data_train = d_tr
        self.data_test = d_te
        self.labels_train = l_tr
        self.labels_test = l_te

        s = range(len(self.data_train))
        d = range(len(self.data_test))
        np.random.shuffle(s)
        np.random.shuffle(d)

        self.s = s
        self.d = d

        # self.__inputs__ = {'data': d_tr, 'labels': l_tr[self.s]}
        # self.__targets__ = {'data': d_te, 'labels': l_te[self.d]}

    def load(self, backend=None, experiment=None):
        # Dataset.inputs

        s = range(len(self.data_train))
        d = range(len(self.data_test))
        np.random.shuffle(s)
        np.random.shuffle(d)

        # self.backend = None
        # self.inputs = {'train': self.data_train[s],
        #                'test': self.data_test[d],
        #                'validation': self.data_test[:5]}
        #
        # self.targets = {'train': self.labels_train[s],
        #                 'test': self.labels_test[d],
        #                 'validation': self.labels_test[:5]}

        self.inputs = {'train': self.data_train[s],
                       'test': self.data_test[d],
                       'validation': None}

        self.targets = {'train': self.labels_train[s],
                        'test': self.labels_test[d],
                        'validation': None}

        # self.original_inputs = self.inputs
        # self.original_targets = self.targets

        self.format()


