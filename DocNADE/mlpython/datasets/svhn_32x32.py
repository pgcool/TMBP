# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

"""
Module ``datasets.svhn_32x32`` gives access to the Street View House Numbers (SVHN) dataset, the 32 x 32 pixels grayscale version.

| **Reference:** 
| Reading Digits in Natural Images with Unsupervised Feature Learning
| Netzer, Wang, Coates, Bissacco, Wu and Ng
| http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os

def load(dir_path,load_to_memory=False):
    """
    Loads the the 32 x 32 pixels version of the Street View House Numbers (SVHN) dataset.

    The original 32 x 32 pixels dataset is in color, but is converted
    in grayscale by this module, in [0,1]. The original training set
    is also split into a new training set and a validation
    set. Finally, the original dataset also includes extra labeled
    examples, which are supposed to be easier to classifier. Those
    were are added to the training set.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'length'``
    * ``'targets'``
    * ``'class_to_id'``

    """

    input_size=1024
    dir_path = os.path.expanduser(dir_path)

    # Put in grayscale, in [0,1]
    def to_grayscale_normalized(example):
        x,y = example
        new_x = (x[:input_size]*0.3 + x[input_size:(2*input_size)]*0.59 + x[(2*input_size):(3*input_size)]*0.11)/255.
        return (new_x,y)

    class TransformedIterator:
        def __init__(self,iter,transform):
            self.iter = iter
            self.transform = transform
            
        def __iter__(self):
            for ex in self.iter:
                yield self.transform(ex)

    if load_to_memory:
        train_inputs = np.load(os.path.join(dir_path,'train_inputs_32x32.npy'))
        valid_inputs = np.load(os.path.join(dir_path,'valid_inputs_32x32.npy'))
        test_inputs = np.load(os.path.join(dir_path,'test_inputs_32x32.npy'))
        train_targets = np.load(os.path.join(dir_path,'train_targets_32x32.npy'))
        valid_targets = np.load(os.path.join(dir_path,'valid_targets_32x32.npy'))
        test_targets = np.load(os.path.join(dir_path,'test_targets_32x32.npy'))


        train = TransformedIterator(mlio.IteratorWithFields(np.hstack([train_inputs,train_targets.reshape(-1,1)]),((0,input_size),(input_size,input_size+1))),
                                    to_grayscale_normalized)
        valid = TransformedIterator(mlio.IteratorWithFields(np.hstack([valid_inputs,valid_targets.reshape(-1,1)]),((0,input_size),(input_size,input_size+1))),
                                    to_grayscale_normalized)
        test = TransformedIterator(mlio.IteratorWithFields(np.hstack([test_inputs,test_targets.reshape(-1,1)]),((0,input_size),(input_size,input_size+1))),
                                   to_grayscale_normalized)

    else:
        def load_line(line):
            tokens = line.split()
            return (np.array([float(i) for i in tokens[:-1]]),int(tokens[-1]))

        train_file,valid_file,test_file = [os.path.join(dir_path, ds + '_32x32.txt') for ds in ['train','valid','test']]
        # Get data
        train,valid,test = [TransformedIterator(mlio.load_from_file(f,load_line),to_grayscale_normalized) for f in [train_file,valid_file,test_file]]

    # Get metadata
    lengths = [594388,10000,26032]
    targets = set(range(1,11))
    class_to_id = {}
    for t in range(10):
        class_to_id[t+1] = t
        
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                                        'length':l,'targets':targets,
                                        'class_to_id':class_to_id} for l in lengths]
    
    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    import scipy.io
    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset (this could take a while)'
    import urllib
    urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat',os.path.join(dir_path,'train_32x32.mat'))
    urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat',os.path.join(dir_path,'test_32x32.mat'))
    urllib.urlretrieve('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',os.path.join(dir_path,'extra_32x32.mat'))
    print 'Extracting the dataset (this could take a while)'
    train_mat = scipy.io.loadmat(os.path.join(dir_path,'train_32x32.mat'))
    test_mat = scipy.io.loadmat(os.path.join(dir_path,'test_32x32.mat'))
    extra_mat = scipy.io.loadmat(os.path.join(dir_path,'extra_32x32.mat'))

    train_inputs = train_mat['X']
    test_inputs = test_mat['X']
    extra_inputs = extra_mat['X']

    train_targets = train_mat['y']
    test_targets = test_mat['y']
    extra_targets = extra_mat['y']

    del train_mat
    del test_mat
    del extra_mat

    # Put in grayscale, in [0,1] and as flat vectors
    def to_new_format(X):
        ret = np.zeros((X.shape[3],32*32*3),dtype='uint8')
        for t in range(X.shape[3]):
            ret[t,:] = np.hstack((X[:,:,0,t].flatten(),X[:,:,1,t].flatten(),X[:,:,2,t].flatten()))
        return ret

    train_inputs = to_new_format(train_inputs)
    test_inputs = to_new_format(test_inputs)
    extra_inputs = to_new_format(extra_inputs)

    n_train_valid = len(train_inputs)
    n_valid = 10000
    n_train = n_train_valid - n_valid

    # Shuffle train data
    import random
    random.seed(25)
    perm = range(n_train_valid)
    random.shuffle(perm)

    valid_inputs = train_inputs[perm[n_train:]]
    train_inputs = train_inputs[perm[:n_train]]

    valid_targets = train_targets[perm[n_train:]]
    train_targets = train_targets[perm[:n_train]]

    # Append extra data
    train_inputs = np.vstack((train_inputs,extra_inputs))
    train_targets = np.vstack((train_targets,extra_targets))

    del extra_inputs
    del extra_targets

    # Reshuffle train
    perm = range(len(train_inputs))
    random.shuffle(perm)
    train_inputs = train_inputs[perm]
    train_targets = train_targets[perm]

    train_targets = train_targets.flatten()
    valid_targets = valid_targets.flatten()
    test_targets = test_targets.flatten()

    def write_to_file(data,labels,file):
        f = open(os.path.join(dir_path,file),'w')
        for input,label in zip(data,labels):
            f.write(' '.join([str(xi) for xi in input]) + ' ' + str(label) + '\n')
        f.close()

    write_to_file(train_inputs,train_targets,'train_32x32.txt')
    write_to_file(valid_inputs,valid_targets,'valid_32x32.txt')
    write_to_file(test_inputs,test_targets,'test_32x32.txt')

    np.save(os.path.join(dir_path,'train_inputs_32x32.npy'),train_inputs)
    np.save(os.path.join(dir_path,'train_targets_32x32.npy'),train_targets)

    np.save(os.path.join(dir_path,'valid_inputs_32x32.npy'),valid_inputs)
    np.save(os.path.join(dir_path,'valid_targets_32x32.npy'),valid_targets)

    np.save(os.path.join(dir_path,'test_inputs_32x32.npy'),test_inputs)
    np.save(os.path.join(dir_path,'test_targets_32x32.npy'),test_targets)

    print 'Done                     '
