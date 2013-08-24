import mlpython.misc.io as mlio
import numpy as np
import os, itertools
from gzip import GzipFile as gfile
import random

def libsvm(input_sparse):
    string = " "
    for i in input_sparse.nonzero()[1]:
        string += "%d:%d " %((i+1), input_sparse[0,i])
    
    string += "\n"
    return string


def load(dir_path,load_to_memory=True,sparse=False):
    """
    Loads the NIPS abstracts dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    The inputs have been put in binary format, and the vocabulary has been
    restricted to 13649 words.

    **Defined metadata:**

    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """

    input_size=13649
    dir_path = os.path.expanduser(dir_path)

    def load_line(line):
        return mlio.libsvm_load_line(line,convert_target=str,sparse=sparse,input_size=input_size,input_type=np.int32)[0]

    # Get data
    train_file,valid_file,test_file = [os.path.join(dir_path, ds + '.txt') for ds in ['train','valid','test']]
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]
    if load_to_memory:
        train,valid,test = [ [x for x in f] for f in [train,valid,test] ]

    lengths = [1640,50,50]

    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size, 'length':l} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}



def obtain(dir_path):
    print("Load nips_data.mat")
    import scipy.io
    mat = scipy.io.loadmat(os.path.join(dir_path,"nips_data.mat"))
    mat_inputs = mat['AA']
    list_index=range(mat_inputs.shape[0])
    #random.seed(1234)
    #random.shuffle(list_index)

    nb_test = 50
    nb_valid = 50
    nb_train = mat_inputs.shape[0] - nb_valid - nb_test
    
    print("create train matrix")
    mat_train_input = mat_inputs[list_index[:nb_train]]

    print("create valid matrix")
    mat_valid_input = mat_inputs[list_index[nb_train:nb_train+nb_valid]]

    print("create test matrix")
    mat_test_input = mat_inputs[list_index[nb_train+nb_valid:]]

    ##save in spase format
    #print("Save train sparse mats")
    #scipy.io.mmwrite(os.path.join(dir_path,'train_input'), mat_train_input)
    #print("Save valid sparse mat")
    #scipy.io.mmwrite(os.path.join(dir_path,'valid_input'), mat_valid_input)
    #print("Save test sparse mat")
    #scipy.io.mmwrite(os.path.join(dir_path,'test_input'), mat_test_input)
    
    # Ecrire fichiers libsvm
    print("Write train.txt files")
    writeFile(os.path.join(dir_path,"train.txt"), mat_train_input)
    print("Write valid.txt files")
    writeFile(os.path.join(dir_path,"valid.txt"), mat_valid_input)
    print("Write test.txt files")
    writeFile(os.path.join(dir_path,"test.txt"), mat_test_input)


def writeFile(fileName, mat_input):
    myfile = open(fileName, "w")
    for row_input in mat_input:
        myfile.write(libsvm(row_input))
    myfile.close()
