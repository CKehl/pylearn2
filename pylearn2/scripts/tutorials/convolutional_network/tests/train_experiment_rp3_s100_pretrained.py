"""
Test that a smaller version of convolutional_network.ipynb works.

The differences (needed for speed) are:
    * output_channels: 4 (64)
    * train.stop: 4000 (40000)
    * valid.stop: 40100 (50000)
    * test.start: 0
    * test.stop: 100 (10000)
    * termination_criterion.max_epochs: 1 (500)

This should make the test run in about one minute.
"""

import os
from pylearn2.testing import skip
from pylearn2.config import yaml_parse
from pylearn2.utils import string_utils
import string

def CNN():
    """Test smaller version of convolutional_network.ipynb"""
    which_experiment = 'RP3_S100'
    skip.skip_if_no_data()
    yaml_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = string_utils.preprocess('${PYLEARN2_DATA_PATH}')
    save_path = os.path.join(data_dir, 'cifar10', 'experiment_'+string.lower(which_experiment))
    base_save_path = os.path.join(data_dir, 'cifar10')
    # Escape potential backslashes in Windows filenames, since
    # they will be processed when the YAML parser will read it
    # as a string
    #save_path.replace('\\', r'\\')

    yaml = open("{0}/experiment_rp3_s100_pretrained.yaml".format(yaml_file_path), 'r').read()
    hyper_params = {'batch_size': 64,
                    'output_channels_h1': 64,
                    'output_channels_h2': 128,
                    'output_channels_h3': 600,
                    'max_epochs': 15,
                    'save_path': save_path,
                    'base_save_path' : base_save_path }
    yaml = yaml % (hyper_params)
    train = yaml_parse.load(yaml)
    train.main_loop()


    #try:
    #    os.remove("{}/convolutional_network_best.pkl".format(save_path))
    #except OSError:
    #    pass

if __name__ == '__main__':
    CNN()
