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
The ``mathutils.topic_modeling`` module contains useful operations
for the learners in mlpython.learners.topic_modeling.

This module defines the following functions:

* ``words_list_from_counts``:               Generates a list of words from sparse document word counts
* ``doc_mnade_tree_fprop_word_probs``:      Computes the binary decision probabilities along the tree path of each words
* ``doc_mnade_tree_bprop_word_probs``:      Computes the gradient of the tree binary decision (log-)probabilities
* ``doc_mnade_tree_update_word_probs``:     Updates the parameters of the tree binary decision parameters
* ``doc_mnade_sparse_update_W``:            Updates matrix W in DocMNADE in a sparse way

"""

import ctypes, os
import numpy as np

# Getting absolute path to this module 
#  (os.abspath() doesn't do the job: it call os.getcwd(), which 
#   gives the path of the executing script that imported this module)
import mlpython
so_path = os.path.join(os.path.dirname(mlpython.__file__),'mathutils/_topic_modeling.so')
_topic_modeling = ctypes.CDLL(so_path)

def words_list_from_counts(word_ids,word_counts):
    """
    Generates a list of words (indices) from the sparse counts of
    words in a document
    """

    assert(len(word_ids.shape) == 1)
    assert(word_ids.shape == word_counts.shape)
    if word_ids.dtype != 'int32':
        word_ids = np.array(word_ids,dtype='int32')
    if word_counts.dtype != 'int32':
        word_counts = np.array(word_counts,dtype='int32')

    words = np.zeros((np.sum(word_counts)),dtype='int32')
    _topic_modeling.words_list_from_counts(word_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                           word_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                           words.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                           word_ids.shape[0])

    return words

def doc_mnade_tree_fprop_word_probs(h,V,b,words,binary_codes,path_lengths,node_ids,tree_probs):
    """
    Computes the binary decision probabilities along the tree path of
    each words.
    """

    # Check arguments sizes
    assert(len(V.shape) == 2)
    assert(V.shape[1] == h.shape[1])
    assert(b.shape[0] == V.shape[0])
    assert(b.shape[0] == binary_codes.shape[0])
    assert(b.shape[0] == path_lengths.shape[0])
    assert(b.shape[0] == node_ids.shape[0])
    assert(h.shape[0] == words.shape[0])
    assert(h.shape[0] == tree_probs.shape[0])
    assert(binary_codes.shape[1] == node_ids.shape[1])
    assert(binary_codes.shape[1] == tree_probs.shape[1])

    # Check argument types
    assert(V.dtype == 'float64' and V.flags['C_CONTIGUOUS'] and V.flags['ALIGNED'])
    assert(h.dtype == 'float64' and h.flags['C_CONTIGUOUS'] and h.flags['ALIGNED'])
    assert(b.dtype == 'float64' and b.flags['C_CONTIGUOUS'] and b.flags['ALIGNED'])
    assert(binary_codes.dtype == 'bool' and binary_codes.flags['C_CONTIGUOUS'] and binary_codes.flags['ALIGNED'])
    assert(path_lengths.dtype == 'int32' and path_lengths.flags['C_CONTIGUOUS'] and path_lengths.flags['ALIGNED'])
    assert(node_ids.dtype == 'int32' and node_ids.flags['C_CONTIGUOUS'] and node_ids.flags['ALIGNED'])
    assert(words.dtype == 'int32' and words.flags['C_CONTIGUOUS'] and words.flags['ALIGNED'])
    assert(tree_probs.dtype == 'float64' and tree_probs.flags['C_CONTIGUOUS'] and tree_probs.flags['ALIGNED'] and tree_probs.flags['WRITEABLE'])

    _topic_modeling.doc_mnade_tree_fprop_word_probs(h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    words.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                    binary_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                    path_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                    node_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                    tree_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    words.shape[0],
                                                    h.shape[1],
                                                    binary_codes.shape[1])

def doc_mnade_tree_bprop_word_probs(h,dh,V,dV,b,db,words,binary_codes,
                                    path_lengths,node_ids,tree_probs,to_update):
    """
    Computes the gradient of the tree binary decision (log-)probabilities.
    """

    # Check arguments sizes
    assert(len(V.shape) == 2)
    assert(V.shape[1] == h.shape[1])
    assert(b.shape[0] == V.shape[0])
    assert(b.shape[0] == binary_codes.shape[0])
    assert(b.shape[0] == path_lengths.shape[0])
    assert(b.shape[0] == node_ids.shape[0])
    assert(h.shape[0] == words.shape[0])
    assert(h.shape[0] == tree_probs.shape[0])
    assert(binary_codes.shape[1] == node_ids.shape[1])
    assert(binary_codes.shape[1] == tree_probs.shape[1])
    assert(dV.shape == V.shape)
    assert(db.shape == b.shape)
    assert(dh.shape == h.shape)
    assert(to_update.shape == path_lengths.shape)

    # Check argument types
    assert(V.dtype == 'float64' and V.flags['C_CONTIGUOUS'] and V.flags['ALIGNED'])
    assert(h.dtype == 'float64' and h.flags['C_CONTIGUOUS'] and h.flags['ALIGNED'])
    assert(b.dtype == 'float64' and b.flags['C_CONTIGUOUS'] and b.flags['ALIGNED'])
    assert(binary_codes.dtype == 'bool' and binary_codes.flags['C_CONTIGUOUS'] and binary_codes.flags['ALIGNED'])
    assert(dV.dtype == 'float64' and dV.flags['C_CONTIGUOUS'] and dV.flags['ALIGNED'] and dV.flags['WRITEABLE'])
    assert(dh.dtype == 'float64' and dh.flags['C_CONTIGUOUS'] and dh.flags['ALIGNED'] and dh.flags['WRITEABLE'])
    assert(db.dtype == 'float64' and db.flags['C_CONTIGUOUS'] and db.flags['ALIGNED'] and db.flags['WRITEABLE'])
    assert(path_lengths.dtype == 'int32' and path_lengths.flags['C_CONTIGUOUS'] and path_lengths.flags['ALIGNED'])
    assert(node_ids.dtype == 'int32' and node_ids.flags['C_CONTIGUOUS'] and node_ids.flags['ALIGNED'])
    assert(words.dtype == 'int32' and words.flags['C_CONTIGUOUS'] and words.flags['ALIGNED'])
    assert(tree_probs.dtype == 'float64' and tree_probs.flags['C_CONTIGUOUS'] and tree_probs.flags['ALIGNED'] and tree_probs.flags['WRITEABLE'])
    assert(to_update.dtype == 'bool' and to_update.flags['C_CONTIGUOUS'] and to_update.flags['ALIGNED'] and to_update.flags['WRITEABLE'])

    _topic_modeling.doc_mnade_tree_bprop_word_probs(h.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    dh.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    dV.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    db.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    words.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                    binary_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                    path_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                    node_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                    tree_probs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                    to_update.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                    words.shape[0],
                                                    h.shape[1],
                                                    node_ids.shape[1])

def doc_mnade_tree_update_word_probs(V,dV,b,db,words,
                                     path_lengths,node_ids,to_update,learning_rate):
    """
    Update the parameters of the tree binary decision logistic regressor.
    It also resets to 0 the gradients used in dV and db.
    """

    # Check arguments sizes
    assert(len(V.shape) == 2)
    assert(b.shape[0] == V.shape[0])
    assert(b.shape[0] == path_lengths.shape[0])
    assert(b.shape[0] == node_ids.shape[0])
    assert(dV.shape == V.shape)
    assert(db.shape == b.shape)
    assert(to_update.shape == path_lengths.shape)

    # Check argument types
    assert(V.dtype == 'float64' and V.flags['C_CONTIGUOUS'] and V.flags['ALIGNED'] and dV.flags['WRITEABLE'])
    assert(b.dtype == 'float64' and b.flags['C_CONTIGUOUS'] and b.flags['ALIGNED'] and db.flags['WRITEABLE'])
    assert(dV.dtype == 'float64' and dV.flags['C_CONTIGUOUS'] and dV.flags['ALIGNED'])
    assert(db.dtype == 'float64' and db.flags['C_CONTIGUOUS'] and db.flags['ALIGNED'])
    assert(node_ids.dtype == 'int32' and node_ids.flags['C_CONTIGUOUS'] and node_ids.flags['ALIGNED'])
    assert(words.dtype == 'int32' and words.flags['C_CONTIGUOUS'] and words.flags['ALIGNED'])
    assert(to_update.dtype == 'bool' and to_update.flags['C_CONTIGUOUS'] and to_update.flags['ALIGNED'] and to_update.flags['WRITEABLE'])

    _topic_modeling.doc_mnade_tree_update_word_probs(V.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                     dV.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                     b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                     db.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                                     words.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                     path_lengths.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                     node_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                                     to_update.ctypes.data_as(ctypes.POINTER(ctypes.c_bool)),
                                                     words.shape[0],
                                                     V.shape[1],
                                                     node_ids.shape[1],
                                                     ctypes.c_double(learning_rate))
    

def doc_mnade_sparse_update_W(W,word_ids,dW,lr):
    """
    Updates matrix W in DocMNADE in a sparse way, based on its
    gradient matrix, the set of words present in the input and the
    learning rate. It also resets to 0 the gradients used in dW.
    """

    assert(len(W.shape) == 2)
    assert(W.shape == dW.shape)
    assert(W.dtype == 'float64')
    assert(dW.dtype == 'float64')
    assert(W.flags['C_CONTIGUOUS'])
    assert(dW.flags['C_CONTIGUOUS'])
    assert(W.flags['ALIGNED'])
    assert(dW.flags['ALIGNED'])
    assert(W.flags['WRITEABLE'])
    assert(dW.flags['WRITEABLE'])

    if word_ids.dtype != 'int32':
        word_ids = np.array(word_ids,dtype='int32')
        
    _topic_modeling.doc_mnade_sparse_update_W(W.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                              word_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                              dW.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                              word_ids.shape[0],
                                              W.shape[1],
                                              ctypes.c_double(lr))
    
