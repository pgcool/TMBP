# Copyright 2011 Hugo larochelle. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY Hugo larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo larochelle.

"""
The ``learners.third_party.rtrank.regression`` module contains 
regression models based on the RT-Rank library:

* RandomForest:          Random forest regression model.
* GradientBoostedTrees:  Ensemble of regression trees trained with gradient boosting.

"""


from mlpython.learners.generic import Learner
import numpy as np

try :
    import pyrtrank
except ImportError:
    import warnings
    warnings.warn('\'import pyrtrank\' failed. The RT-Rank library is not properly installed. See mlpython/learners/third_party/rtrank/README for instructions.')

class RandomForest(Learner):
    """ 
    Random Forest regression model based on the RT-Rank decision tree
    learning library
 
    Option ``n_trees`` is the number of trees to train in the ensemble
    (default = 50).

    Option ``n_features_per_node`` is the number of inputs (features)
    to consider when splitting a tree node. The default (None) is
    to use the square root of the input size.

    Option ``max_height`` is the maximum height of the trees.
    The default (None) puts no limit on the tree size. A single
    root node is considered to be of height 1.

    Option ``zeros_are_missing`` is True if it should be considered
    as a missing value which will be split into a special "MISSING"
    node. Otherwise, treat zeros normally (default).

    Option ``seed`` is the seed of the random number generator.

    **Required metadata:**

    * ``'input_size'``

    """
    def __init__(self, n_trees = 50, 
                 n_features_per_node = None, 
                 max_height = None,
                 zeros_are_missing = False,
                 seed = 1234):
        self.n_trees = n_trees
        self.n_features_per_node = n_features_per_node
        self.max_height = max_height
        self.zeros_are_missing = zeros_are_missing
        self.seed = seed

    def train(self,trainset):
        """
        Trains a random forest using RT-Rank.
        """

        self.input_size = trainset.metadata['input_size']
        if self.max_height is None or self.max_height < 1:
            maxdepth = -1
        else:
            maxdepth = self.max_height

        if self.n_features_per_node is None:
            k_random_features = int(np.round(np.sqrt(self.input_size)))
        else:
            k_random_features = self.n_features_per_node

        self.trees = pyrtrank.pyrtrank_interface(self.input_size,
                                                 maxdepth,
                                                 k_random_features,
                                                 self.seed)

        # Add examples to RT-Rank training set
        import sys
        print 'Passing training data to RT-Rank library'
        sys.stdout.flush()
        for x,y in trainset:
            self.trees.add_example(float(y),1,[xi for xi in x])

        print 'Training decision trees'
        sys.stdout.flush()
        for t in range(self.n_trees):
            # Add trees one at a time
            self.trees.train(False,False,True,self.zeros_are_missing,1)

    def use(self,dataset):
        """
        Outputs the target predictions for ``dataset``.
        """
        outputs = np.zeros((len(dataset),1))
        t = 0
        for x,y in dataset:
            outputs[t,0] = 0
            x_list = [xi for xi in x]
            for tree in range(self.n_trees):
                outputs[t,0] += self.trees.predict(tree,x_list)
            outputs[t,0] /= self.n_trees
            t+=1
        return outputs

    def forget(self):
        self.trees = None

    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
        the regression error cost for each example in the dataset.
        """
        outputs = self.use(dataset)
        
        costs = np.ones((len(outputs),1))
        # Compute mean squared error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            cost[0] = (y-pred)**2

        return outputs,costs


class GradientBoostedTrees(Learner):
    """ 
    Ensemble of regression trees trained wiht gradient boosting
 
    Option ``n_trees`` is the number of trees to train in the ensemble
    (default = 50).

    Option ``n_features_per_node`` is the number of inputs (features)
    to consider when splitting a tree node. The default (None) is
    to use the square root of the input size.

    Option ``max_height`` is the maximum height of the trees.
    The default (None) puts no limit on the tree size. A single
    root node is considered to be of height 1.

    Option ``learning_rate`` is the gradient boosting learning rate
    (default = 0.1).

    Option ``zeros_are_missing`` is True if it should be considered
    as a missing value which will be split into a special "MISSING"
    node. Otherwise, treat zeros normally (default).

    Option ``n_processors`` is the number of processors to use
    for growing trees (default = 1).

    Option ``seed`` is the seed of the random number generator.

    **Required metadata:**

    * ``'input_size'``

    """
    def __init__(self, n_trees = 50, 
                 n_features_per_node = None, 
                 max_height = None, 
                 learning_rate = 0.1,
                 zeros_are_missing = False,
                 n_processors = 1,
                 seed = 1234):
        self.n_trees = n_trees
        self.n_features_per_node = n_features_per_node
        self.max_height = max_height
        self.learning_rate = learning_rate
        self.zeros_are_missing = zeros_are_missing
        self.n_processors = n_processors
        self.seed = seed        

    def train(self,trainset):
        """
        Trains an ensemble of trees using gradient boosting and RT-Rank.
        """

        self.input_size = trainset.metadata['input_size']
        if self.max_height is None or self.max_height < 1:
            maxdepth = -1
        else:
            maxdepth = self.max_height

        if self.n_features_per_node is None:
            k_random_features = int(np.round(np.sqrt(self.input_size)))
        else:
            k_random_features = self.n_features_per_node

        self.trees = pyrtrank.pyrtrank_interface(self.input_size,
                                                 maxdepth,
                                                 k_random_features,
                                                 self.seed)

        # Add examples to RT-Rank training set
        import sys
        print 'Passing training data to RT-Rank library'
        sys.stdout.flush()
        for x,y in trainset:
            self.trees.add_example(float(y),1,[xi for xi in x])

        print 'Training decision trees'
        sys.stdout.flush()
        for tree in range(self.n_trees):
            # Add trees one at a time
            self.trees.train(False,False,False,self.zeros_are_missing,self.n_processors)
            self.trees.subtract_predictions_from_targets(tree,self.learning_rate)

    def use(self,dataset):
        """
        Outputs the target predictions for ``dataset``.
        """
        outputs = np.zeros((len(dataset),1))
        t = 0
        for x,y in dataset:
            outputs[t,0] = 0
            x_list = [xi for xi in x]
            for tree in range(self.n_trees):
                outputs[t,0] += self.learning_rate * self.trees.predict(tree,x_list)
            t+=1
        return outputs

    def forget(self):
        self.trees = None

    def test(self,dataset):
        """
        Outputs the result of ``use(dataset)`` and 
        the regression error cost for each example in the dataset.
        """
        outputs = self.use(dataset)
        
        costs = np.ones((len(outputs),1))
        # Compute mean squared error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            cost[0] = (y-pred)**2

        return outputs,costs
