"""
.. todo::

    WRITEME
"""
import os
import logging
import string

import numpy
#N = np = numpy
from theano.compat.six.moves import xrange

from pylearn2.datasets import cache, dense_design_matrix
from pylearn2.expr.preprocessing import global_contrast_normalize
from pylearn2.utils import contains_nan
from pylearn2.utils import serial
from pylearn2.utils import string_utils


_logger = logging.getLogger(__name__)


class Experiment(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : str
        One of 'train', 'test'
    which_experiment : str
        One of 'S100', 'ADD3_10_S100', 'ADD3_10_S250', 'ADD3_ALL_S100', 'RM3_S100' and 'RP3_S100'
    center : WRITEME
    rescale : WRITEME
    gcn : float, optional
        Multiplicative constant to use for global contrast normalization.
        No global contrast normalization is applied, if None
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, which_set, which_experiment, center=False, gcn=None,
                 start=None, stop=None, axes=('b', 0, 1, 'c'),
                 toronto_prepro = False, preprocessor = None):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)
        assert which_set in ['train', 'test','valid']
        assert which_experiment in ['S100', 'ADD3_10_S100', 'ADD3_10_S250', 'ADD3_ALL_S100', 'RM3_S100', 'RP3_S100']
        
        self.experiment = which_experiment

        index_set = which_set
        if index_set == 'valid':
            index_set = 'train'
        data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')
        experiment_folder_string = "experiment_"+string.lower(which_experiment)
        path = os.path.join(data_dir,"cifar10",experiment_folder_string,index_set)
        meta_path = os.path.join(data_dir,"cifar10",experiment_folder_string,"meta")
        

        self.axes = axes

        # we also expose the following details:
        self.img_shape = (3, 32, 32)
        self.img_size = numpy.prod(self.img_shape)
        
        
        meta = serial.load(meta_path)
        #self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.label_names = meta['label_names']
        self.n_classes = len(self.label_names)

        

        obj = serial.load(path)
        X = obj['data']
            

        assert X.max() == 255.
        assert X.min() == 0.

        X = numpy.cast['float32'](X)
        y = numpy.zeros((X.shape[0], 1), dtype=numpy.uint8)
        y[0:X.shape[0],0] = numpy.asarray(obj['labels']).astype(numpy.uint8)
        #y = numpy.asarray(obj['labels']).astype(numpy.uint8)
        
        if(which_set == 'train'):
            ntrain = X.shape[0]
        if(which_set == 'test'):
            ntest = X.shape[0]
        if(which_set == 'valid'):
            iarray = numpy.random.randint(X.shape[0], size=1000)
            X = X[iarray]
            y = y[iarray]
        
        assert X.shape[0] == y.shape[0]

        #y_s = numpy.asarray(obj['labels']).astype(numpy.uint8)
        #y = numpy.zeros((y_s.shape[0], self.n_classes), dtype=numpy.uint8)
        #for i in xrange(y_s.shape[0]):
        #    label = y_s[i]
        #    y[i,label]=1.0


        if center:
            X -= 127.5
        self.center = center

        if toronto_prepro:
            assert not center
            assert not gcn
            if which_set == 'test':
                raise NotImplementedError("Need to subtract the mean of the "
                                          "*training* set.")
            X = X / 255.
            X = X - X.mean(axis=0)
        self.toronto_prepro = toronto_prepro

        self.gcn = gcn
        if gcn is not None:
            assert isinstance(gcn, float)
            X = (X.T - X.mean(axis=1)).T
            X = (X.T / numpy.sqrt(numpy.square(X).sum(axis=1))).T
            X *= gcn

        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop, :]
        

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3), axes)

        super(Experiment, self).__init__(X=X, y=y, y_labels=self.n_classes, view_converter=view_converter, axes=self.axes)

        assert not contains_nan(self.X)

        #if preprocessor:
        #    preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the new
        # ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            for i in xrange(rval.shape[0]):
                rval[i, :] /= numpy.abs(rval[i, :]).max()
            return rval

        if not self.center:
            rval -= 127.5

        rval /= 127.5
        rval = numpy.clip(rval, -1., 1.)

        return rval

    def __setstate__(self, state):
        super(Experiment, self).__setstate__(state)
        # Patch old pkls
        if self.y is not None and self.y.ndim == 1:
            self.y = self.y.reshape((self.y.shape[0], 1))
        #if 'y_labels' not in state:
        #    self.y_labels = 10
        assert 'y_labels' in state

    def adjust_to_be_viewed_with(self, X, orig, per_example=False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the scale
        # determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new
        # ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        if self.gcn is not None:
            rval = X.copy()
            if per_example:
                for i in xrange(rval.shape[0]):
                    rval[i, :] /= numpy.abs(orig[i, :]).max()
            else:
                rval /= numpy.abs(orig).max()
            rval = numpy.clip(rval, -1., 1.)
            return rval

        if not self.center:
            rval -= 127.5

        rval /= 127.5
        rval = numpy.clip(rval, -1., 1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return Experiment(which_set='test', center=self.center,
                       gcn=self.gcn,
                       toronto_prepro=self.toronto_prepro,
                       axes=self.axes)

class Experiment_extended(dense_design_matrix.DenseDesignMatrix):

    """
    .. todo::

        WRITEME

    Parameters
    ----------
    which_set : str
        One of 'train', 'test'
    which_experiment : str
        One of 'S100', 'ADD3_10_S100', 'ADD3_10_S250', 'ADD3_ALL_S100', 'RM3_S100' and 'RP3_S100'
    center : WRITEME
    rescale : WRITEME
    gcn : float, optional
        Multiplicative constant to use for global contrast normalization.
        No global contrast normalization is applied, if None
    start : WRITEME
    stop : WRITEME
    axes : WRITEME
    toronto_prepro : WRITEME
    preprocessor : WRITEME
    """

    def __init__(self, which_set, which_experiment, start=None, stop=None, axes=('b', 0, 1, 'c'), preprocessor = None):
        # note: there is no such thing as the cifar10 validation set;
        # pylearn1 defined one but really it should be user-configurable
        # (as it is here)
        assert which_set in ['train', 'test']
        assert which_experiment in ['S100', 'ADD3_10_S100', 'ADD3_10_S250', 'ADD3_ALL_S100', 'RM3_S100', 'RP3_S100']
        self.experiment = which_experiment

        data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')
        experiment_folder_string = "experiment_"+string.lower(which_experiment)
        path = os.path.join(data_dir,"cifar10",experiment_folder_string,which_set+".pkl")
        meta_path = os.path.join(data_dir,"cifar10",experiment_folder_string,"meta")
        

        self.axes = axes

        # we also expose the following details:
        self.img_shape = (3, 32, 32)
        self.img_size = numpy.prod(self.img_shape)
        
        
        meta = serial.load(meta_path)
        #self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.label_names = meta['label_names']
        self.n_classes = len(self.label_names)

        

        obj = serial.load(path)
        X = obj['data']
        if(which_set == 'train'):
            ntrain = X.shape[0]
        if(which_set == 'test'):
            ntest = X.shape[0]

        assert X.max() == 255.
        assert X.min() == 0.

        X = numpy.cast['float32'](X)
        y = numpy.asarray(obj['labels']).astype('uint8')
        
        if which_set == 'test':
            y = y.reshape((y.shape[0], 1))

        if start is not None:
            # This needs to come after the prepro so that it doesn't
            # change the pixel means computed above for toronto_prepro
            assert start >= 0
            assert stop > start
            assert stop <= X.shape[0]
            X = X[start:stop, :]
            y = y[start:stop, :]
        assert X.shape[0] == y.shape[0]

        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 3), axes)

        super(Experiment, self).__init__(X=X, y=y, y_labels=self.n_classes, view_converter=view_converter, axes=self.axes)

        assert not contains_nan(self.X)

        #if preprocessor:
        #    preprocessor.apply(self)

    def adjust_for_viewer(self, X):
        """
        .. todo::

            WRITEME
        """
        # assumes no preprocessing. need to make preprocessors mark the new
        # ranges
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        rval /= 127.5
        rval = numpy.clip(rval, -1., 1.)

        return rval

    def __setstate__(self, state):
        super(Experiment, self).__setstate__(state)
        # Patch old pkls
        if self.y is not None and self.y.ndim == 1:
            self.y = self.y.reshape((self.y.shape[0], 1))
        #if 'y_labels' not in state:
        #    self.y_labels = 10
        assert 'y_labels' in state
        assert hasattr(self, 'experiment')

    def adjust_to_be_viewed_with(self, X, orig, per_example=False):
        """
        .. todo::

            WRITEME
        """
        # if the scale is set based on the data, display X oring the scale
        # determined
        # by orig
        # assumes no preprocessing. need to make preprocessors mark the new
        # ranges
        assert hasattr(self, 'experiment')
        rval = X.copy()

        # patch old pkl files
        if not hasattr(self, 'center'):
            self.center = False
        if not hasattr(self, 'gcn'):
            self.gcn = False

        rval /= 127.5
        rval = numpy.clip(rval, -1., 1.)

        return rval

    def get_test_set(self):
        """
        .. todo::

            WRITEME
        """
        return Experiment(which_set='test', which_experiment=self.experiment, axes=self.axes)



