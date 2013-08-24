import mlpython.misc.io as mlio
import numpy as np
import os, itertools
from gzip import GzipFile as gfile
import random

def libsvm(input_sparse, target):
    
    string = "%d " %target
    
    for i in input_sparse.nonzero()[1]:
        string += "%d:%d " %((i+1), input_sparse[0,i])
    
    string += "\n"
    return string


def load(dir_path,load_to_memory=True,sparse=False,binary_input=False):
    """
    Loads the 20 news groups dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The inputs have been put in binary format, and the vocabulary has been
    restricted to 2000 words.

    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """

    input_size=2000
    targets = set(range(20))
    dir_path = os.path.expanduser(dir_path)
    def convert_target(target_str):
        return int(target_str)-1
        
    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=convert_target,sparse=sparse,input_size=input_size,input_type=np.int32)

    # Get data
    train_file,valid_file,test_file = [os.path.join(dir_path, ds + '.txt') for ds in ['train','valid','test']]
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    if load_to_memory:
        train,valid,test = [ [x for x in f] for f in [train,valid,test] ]

    lengths = [10284,1000,7502]

    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,'targets':targets,'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}



def obtain(dir_path):
    print("Load 20news_data.mat")
    import scipy.io
    mat = scipy.io.loadmat(os.path.join(dir_path,"20news_data.mat"))
    mat_sparse_train_inputs = mat['AA2000_train']
    vec_dense_train_targets = mat['target'][0]
    mat_sparse_test_inputs = mat['AA2000_test']
    vec_dense_test_targets = mat['target_test'][0]
    list_index_train=range(mat_sparse_train_inputs.shape[0])
    list_index_test=range(mat_sparse_test_inputs.shape[0])
    # TODO: seter le seed
    random.seed(1234)
    random.shuffle(list_index_train)
    #random.shuffle(list_index_test)

    nb_test = mat_sparse_test_inputs.shape[0]
    nb_valid = 1000
    nb_train = mat_sparse_train_inputs.shape[0] - nb_valid
    
    print("create train matrix")
    mat_train_input = mat_sparse_train_inputs[list_index_train[:nb_train]]
    vec_train_target = vec_dense_train_targets[list_index_train[:nb_train]]

    print("create valid matrix")
    mat_valid_input = mat_sparse_train_inputs[list_index_train[nb_train:nb_train+nb_valid]]
    vec_valid_target = vec_dense_train_targets[list_index_train[nb_train:nb_train+nb_valid]]

    print("create test matrix")
    mat_test_input = mat_sparse_test_inputs[list_index_test]
    vec_test_target = vec_dense_test_targets[list_index_test]

    ##save in spase format
    #print("Save train sparse mats")
    #scipy.io.mmwrite('train_input', mat_train_input)
    #scipy.io.mmwrite('train_target',mat_train_target)
    #print("Save valid sparse mat")
    #scipy.io.mmwrite('valid_input', mat_valid_input)
    #scipy.io.mmwrite('valid_target', mat_valid_target)
    #print("Save test sparse mat")
    #scipy.io.mmwrite('test_input', mat_test_input)
    #scipy.io.mmwrite('test_target', mat_test_target)
    
    # Ecrire fichiers libsvm
    print("Write train.txt files")
    writeFile(os.path.join(dir_path,"train.txt"), mat_train_input, vec_train_target)
    print("Write valid.txt files")
    writeFile(os.path.join(dir_path,"valid.txt"), mat_valid_input, vec_valid_target)
    print("Write test.txt files")
    writeFile(os.path.join(dir_path,"test.txt"), mat_test_input, vec_test_target)


def writeFile(fileName, mat_input, vec_target):
    myfile = open(fileName, "w")
    for row_input, target in itertools.izip(mat_input, vec_target):
        myfile.write(libsvm(row_input, target))
    myfile.close()
