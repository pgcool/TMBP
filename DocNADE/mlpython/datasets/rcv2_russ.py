import mlpython.misc.io as mlio
import numpy as np
import os, itertools
from gzip import GzipFile as gfile
import random

def libsvm(input_sparse, target_sparse):
    string = ""
    for i in target_sparse.nonzero()[1]:
        string += "%d," %i
    
    string = string[:-1]
    string += " "
    for i in input_sparse.nonzero()[1]:
        string += "%d:%d " %((i+1), input_sparse[0,i])
    
    string += "\n"
    return string


def load(dir_path,load_to_memory=True,sparse=False,binary_input=False):
    """
    Loads the RCV2 dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The inputs have been put in binary format, and the vocabulary has been
    restricted to 10000 words.

    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """

    input_size=10000
    target_size=103
    dir_path = os.path.expanduser(dir_path)
    def convert_target(target_str):
        targets = np.zeros((target_size))
        for l in target_str.split(','):
            id = int(l)
            targets[id] = 1
        return targets

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=sparse,input_size=input_size,input_type=np.int32)

    # Get data
    train_file,valid_file,test_file = [os.path.join(dir_path, ds + '.txt') for ds in ['train','valid','test']]
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    if load_to_memory:
        train,valid,test = [ [x for x in f] for f in [train,valid,test] ]

    #lengths = [784414,10000,10000]
    lengths = [392207,10000,402207]

    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,'targets':set(range(2)),'target_size':target_size,'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}



def obtain(dir_path):
    #print("Load rcv2_data.mat")
    #import scipy.io
    #mat = scipy.io.loadmat(os.path.join(dir_path,"rcv2_data.mat"))
    #mat_inputs = mat['AA']
    #mat_targets = mat['targets']
    #list_index=range(mat_inputs.shape[0])
    # TODO: seter le seed
    #random.shuffle(list_index)
    #
    #nb_test = 10000
    #nb_valid = 10000
    #nb_train = mat_inputs.shape[0] - nb_valid - nb_test

    print("Load reut_data.mat")
    import scipy.io
    mat = scipy.io.loadmat(os.path.join(dir_path,"reut_data.mat"))
    mat_sparse_train_inputs = mat['AA10000_train']
    mat_dense_train_targets = mat['targets_train']
    mat_sparse_test_inputs = mat['AA10000_test']
    mat_dense_test_targets = mat['targets_test']
    list_index_train=range(mat_sparse_train_inputs.shape[0])
    list_index_test=range(mat_sparse_test_inputs.shape[0])
    # TODO: seter le seed
    random.seed(1234)
    random.shuffle(list_index_train)
    #random.shuffle(list_index_test)
    nb_test = mat_sparse_test_inputs.shape[0]
    nb_valid = 10000
    nb_train = mat_sparse_train_inputs.shape[0] - nb_valid

    print("create train matrix")
    mat_train_input = mat_sparse_train_inputs[list_index_train[:nb_train]]
    mat_train_target = mat_dense_train_targets[list_index_train[:nb_train]]

    print("create valid matrix")
    mat_valid_input = mat_sparse_train_inputs[list_index_train[nb_train:nb_train+nb_valid]]
    mat_valid_target = mat_dense_train_targets[list_index_train[nb_train:nb_train+nb_valid]]

    print("create test matrix")
    mat_test_input = mat_sparse_test_inputs[list_index_test]
    mat_test_target = mat_dense_test_targets[list_index_test]

    ##save in spase format
    #print("Save train sparse mats")
    #scipy.io.mmwrite(os.path.join(dir_path,'train_input'), mat_train_input)
    #scipy.io.mmwrite(os.path.join(dir_path,'train_target'),mat_train_target)
    #print("Save valid sparse mat")
    #scipy.io.mmwrite(os.path.join(dir_path,'valid_input'), mat_valid_input)
    #scipy.io.mmwrite(os.path.join(dir_path,'valid_target'), mat_valid_target)
    #print("Save test sparse mat")
    #scipy.io.mmwrite(os.path.join(dir_path,'test_input'), mat_test_input)
    #scipy.io.mmwrite(os.path.join(dir_path,'test_target'), mat_test_target)
    
    # Ecrire fichiers libsvm
    print("Write train.txt files (this will take a little while)")
    writeFile(os.path.join(dir_path,"train.txt"), mat_train_input, mat_train_target)
    print("Write valid.txt files")
    writeFile(os.path.join(dir_path,"valid.txt"), mat_valid_input, mat_valid_target)
    print("Write test.txt files (this will take a little while)")
    writeFile(os.path.join(dir_path,"test.txt"), mat_test_input, mat_test_target)


def writeFile(fileName, mat_input, mat_target):
    myfile = open(fileName, "w")
    for row_input, row_target in itertools.izip(mat_input, mat_target):
        myfile.write(libsvm(row_input, row_target))
    myfile.close()
