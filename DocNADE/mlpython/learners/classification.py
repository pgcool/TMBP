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
The ``learners.classification`` module contains Learners meant for classification problems. 
They normally will require (at least) the metadata ``'targets'``.
The MLProblems for these Learners should be iterators over pairs
of inputs and targets, with the target being a class index.

The currently implemented algorithms are:

* BayesClassifier:       Bayes classifier obtained from distribution estimators.
* NNet:                  Neural Network for classification.
* ClusteredWeightsNNet:  Neural Network with cluster-dependent weights, for classification.

"""

from generic import Learner,OnlineLearner
import numpy as np
import mlpython.mlproblems.classification as mlpb
import mlpython.mlproblems.generic as mlpbgen
import mlpython.learners.features as mlfeat
import mlpython.mathutils.nonlinear as mlnonlin
import mlpython.mathutils.linalg as mllin


class BayesClassifier(Learner):
    """ 
    Bayes classifier from distribution estimators.
 
    Given one distribution learner per class (option ``estimators``), this
    learner will train each one on a separate class and classify
    examples using Bayes' rule.

    **Required metadata:**
    
    * ``'targets'``

    """
    def __init__(self,
                 estimators=[],# The distribution learners to be trained
                 ):
        self.stage = 0
        self.estimators = estimators

    def train(self,trainset):
        """
        Trains each estimator. Each call to train increments ``self.stage`` by 1.
        If ``self.stage == 0``, first initialize the model.
        """

        self.n_classes = len(trainset.metadata['targets'])

        # Initialize model
        if self.stage == 0:
            # Split data according to classes
            self.class_trainset = []
            tot_len = len(trainset)
            self.prior = np.zeros((self.n_classes))
            for c in xrange(self.n_classes):
                trainset_c = mlpb.ClassSubsetProblem(data=trainset,metadata=trainset.metadata,
                                                     subset=set([c]),
                                                     include_class=False)
                trainset_c.setup()
                self.class_trainset += [ trainset_c ]
                self.prior[c] = float(len(trainset_c))/tot_len

        # Training each estimators
        for c in xrange(self.n_classes):
            self.estimators[c].train(self.class_trainset[c])
        self.stage += 1

    def forget(self):
        self.stage = 0 # Model will be untrained after initialization
        # Initialize estimators
        for c in xrange(self.n_classes):
            self.estimators[c].forget()
        self.prior = 1./self.n_classes * np.ones((self.n_classes))

    def use(self,dataset):
        """
        Outputs the class_id chosen by the algorithm, for each
        example in the dataset.
        """
        outputs = -1*np.ones((len(dataset),1))
        for xy,pred in zip(dataset,outputs):
            x,y = xy
            max_prob = -np.inf
            max_prob_class = -1
            for c in xrange(self.n_classes):
                prob_c = self.estimators[c].use([x])[0] + np.log(self.prior[c])
                if max_prob < prob_c:
                    max_prob = prob_c
                    max_prob_class = c
                
            pred[0] = max_prob_class
            
        return outputs

    def test(self,dataset):
        """
        Outputs the class_id chosen by the algorithm and
        the classification error cost for each example in the dataset
        """
        outputs = self.use(dataset)
        costs = np.ones((len(outputs),1))
        # Compute classification error
        for xy,pred,cost in zip(dataset,outputs,costs):
            x,y = xy
            if y == pred[0]:
                cost[0] = 0

        return outputs,costs


class NNet(OnlineLearner):
   """
   Neural Network for classification.

   Option ``n_stages`` is the number of training iterations.

   Options ``learning_rate`` and ``decrease_constant`` correspond
   to the learning rate and decrease constant used for stochastic
   gradient descent.

   Option ``hidden_sizes`` should be a list of positive integers
   specifying the number of hidden units in each hidden layer, from
   the first to the last.
   
   Option ``activation_function`` should be string describing
   the hidden unit activation function to use. Choices are
   ``'sigmoid'`` (default), ``'tanh'`` and ``'reclin'``.
   
   Option ``seed`` determines the seed for randomly initializing the
   weights.

   Option ``pretrained_parameters`` should be a pair made of the
   list of hidden layer weights and biases, to replace random
   initialization. If None (default), random initialization will
   be used.

   **Required metadata:**

   * ``'input_size'``: Size of the input.
   * ``'targets'``: Set of possible targets.

   """

   def __init__(self, n_stages, 
                learning_rate = 0.01, 
                decrease_constant = 0,
                hidden_sizes = [ 100 ],
                activation_function = 'sigmoid',
                seed = 1234,
                pretrained_parameters = None
                ):
       self.n_stages = n_stages
       self.stage = 0
       self.learning_rate = learning_rate
       self.decrease_constant = decrease_constant
       self.hidden_sizes = hidden_sizes
       self.activation_function = activation_function
       if self.activation_function not in ['sigmoid','tanh','reclin']:
           raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

       self.seed = seed
       self.pretrained_parameters = pretrained_parameters

   def initialize_learner(self,metadata):
      self.n_classes = len(metadata['targets'])
      self.rng = np.random.mtrand.RandomState(self.seed)
      self.input_size = metadata['input_size']
      self.n_hidden_layers = len(self.hidden_sizes)
      if sum([nhid > 0 for nhid in self.hidden_sizes]) != self.n_hidden_layers:
         raise ValueError('All hidden layer sizes should be > 0')
      if self.n_hidden_layers < 1:
         raise ValueError('There should be at least one hidden layer')

      self.Ws = [(2*self.rng.rand(self.hidden_sizes[0],self.input_size)-1)/self.input_size]
      self.cs = [np.zeros((self.hidden_sizes[0]))]
      self.dWs = [np.zeros((self.hidden_sizes[0],self.input_size))]
      self.dcs = [np.zeros((self.hidden_sizes[0]))]

      self.layers = [np.zeros((self.input_size))]
      self.layer_acts = [np.zeros((self.input_size))]
      self.layers += [np.zeros((self.hidden_sizes[0]))]
      self.layer_acts += [np.zeros((self.hidden_sizes[0]))]

      self.dlayers = [np.zeros((self.input_size))]
      self.dlayer_acts = [np.zeros((self.input_size))]
      self.dlayers += [np.zeros((self.hidden_sizes[0]))]
      self.dlayer_acts += [np.zeros((self.hidden_sizes[0]))]

      for h in range(1,self.n_hidden_layers):
         self.Ws += [(2*self.rng.rand(self.hidden_sizes[h],self.hidden_sizes[h-1])-1)/self.hidden_sizes[h-1]]
         self.cs += [np.zeros((self.hidden_sizes[h]))]
         self.dWs += [np.zeros((self.hidden_sizes[h],self.hidden_sizes[h-1]))]
         self.dcs += [np.zeros((self.hidden_sizes[h]))]
         self.layers += [np.zeros((self.hidden_sizes[h]))]
         self.layer_acts += [np.zeros((self.hidden_sizes[h]))]
         self.dlayers += [np.zeros((self.hidden_sizes[h]))]
         self.dlayer_acts += [np.zeros((self.hidden_sizes[h]))]

      self.U = (2*self.rng.rand(self.n_classes,self.hidden_sizes[-1])-1)/self.hidden_sizes[-1]
      self.d = np.zeros((self.n_classes))
      self.dU = np.zeros((self.n_classes,self.hidden_sizes[-1]))
      self.dd = np.zeros((self.n_classes))
      self.output_act = np.zeros((self.n_classes))
      self.output = np.zeros((self.n_classes))
      self.doutput_act = np.zeros((self.n_classes))

      if self.pretrained_parameters is not None:
         self.Ws = self.pretrained_parameters[0]
         self.cs = self.pretrained_parameters[1]

      self.n_updates = 0

   def update_learner(self,example):
      self.layers[0][:] = example[0]

      # fprop
      for h in range(self.n_hidden_layers):
         mllin.product_matrix_vector(self.Ws[h],self.layers[h],self.layer_acts[h+1])
         self.layer_acts[h+1] += self.cs[h]
         if self.activation_function == 'sigmoid':
             mlnonlin.sigmoid(self.layer_acts[h+1],self.layers[h+1])
         elif self.activation_function == 'tanh':
             mlnonlin.tanh(self.layer_acts[h+1],self.layers[h+1])
         elif self.activation_function == 'reclin':
             mlnonlin.reclin(self.layer_acts[h+1],self.layers[h+1])
         else:
             raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

      mllin.product_matrix_vector(self.U,self.layers[-1],self.output_act)
      self.output_act += self.d
      mlnonlin.softmax(self.output_act,self.output)

      self.doutput_act[:] = self.output
      self.doutput_act[example[1]] -= 1
      self.doutput_act *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)

      self.dd[:] = self.doutput_act
      mllin.outer(self.doutput_act,self.layers[-1],self.dU)      
      mllin.product_matrix_vector(self.U.T,self.doutput_act,self.dlayers[-1])
      if self.activation_function == 'sigmoid':
          mlnonlin.dsigmoid(self.layers[-1],self.dlayers[-1],self.dlayer_acts[-1])
      elif self.activation_function == 'tanh':
          mlnonlin.dtanh(self.layers[-1],self.dlayers[-1],self.dlayer_acts[-1])
      elif self.activation_function == 'reclin':
          mlnonlin.dreclin(self.layers[-1],self.dlayers[-1],self.dlayer_acts[-1])
      else:
          raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

      for h in range(self.n_hidden_layers-1,-1,-1):
         self.dcs[h][:] = self.dlayer_acts[h+1]
         mllin.outer(self.dlayer_acts[h+1],self.layers[h],self.dWs[h])
         mllin.product_matrix_vector(self.Ws[h].T,self.dlayer_acts[h+1],self.dlayers[h])
         if self.activation_function == 'sigmoid':
             mlnonlin.dsigmoid(self.layers[h],self.dlayers[h],self.dlayer_acts[h])
         elif self.activation_function == 'tanh':
             mlnonlin.dtanh(self.layers[h],self.dlayers[h],self.dlayer_acts[h])
         elif self.activation_function == 'reclin':
             mlnonlin.dreclin(self.layers[h],self.dlayers[h],self.dlayer_acts[h])
         else:
             raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

      self.U -= self.dU
      self.d -= self.dd
      for h in range(self.n_hidden_layers-1,-1,-1):
         self.Ws[h] -= self.dWs[h]
         self.cs[h] -= self.dcs[h]

      self.n_updates += 1

   def use_learner(self,example):
      output = np.zeros((self.n_classes))
      self.layers[0][:] = example[0]

      # fprop
      for h in range(self.n_hidden_layers):
         mllin.product_matrix_vector(self.Ws[h],self.layers[h],self.layer_acts[h+1])
         self.layer_acts[h+1] += self.cs[h]
         if self.activation_function == 'sigmoid':
             mlnonlin.sigmoid(self.layer_acts[h+1],self.layers[h+1])
         elif self.activation_function == 'tanh':
             mlnonlin.tanh(self.layer_acts[h+1],self.layers[h+1])
         elif self.activation_function == 'reclin':
             mlnonlin.reclin(self.layer_acts[h+1],self.layers[h+1])
         else:
             raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

      mllin.product_matrix_vector(self.U,self.layers[-1],self.output_act)
      self.output_act += self.d
      mlnonlin.softmax(self.output_act,output)

      return [output.argmax(),output]

   def cost(self,outputs,example):
      target = example[1]
      class_id,output = outputs

      return [ target!=class_id,-np.log(output[target])]

   def verify_gradients(self):
      
      print 'WARNING: calling verify_gradients reinitializes the learner'

      rng = np.random.mtrand.RandomState(1234)
      input_order = range(20)
      rng.shuffle(input_order)

      self.seed = 1234
      self.hidden_sizes = [4,5,6]
      self.initialize_learner({'input_size':20,'targets':set([0,1,2])})
      example = (rng.rand(20)<0.5,2)
      epsilon=1e-6
      self.learning_rate = 1
      self.decrease_constant = 0

      import copy
      Ws_copy = copy.deepcopy(self.Ws)
      emp_dWs = copy.deepcopy(self.Ws)
      for h in range(self.n_hidden_layers):
         for i in range(self.Ws[h].shape[0]):
            for j in range(self.Ws[h].shape[1]):
               self.Ws[h][i,j] += epsilon
               output = self.use_learner(example)
               a = self.cost(output,example)[1]
               self.Ws[h][i,j] -= epsilon
               
               self.Ws[h][i,j] -= epsilon
               output = self.use_learner(example)
               b = self.cost(output,example)[1]
               self.Ws[h][i,j] += epsilon

               emp_dWs[h][i,j] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.Ws = Ws_copy
      print 'dWs[0] diff.:',np.sum(np.abs(self.dWs[0].ravel()-emp_dWs[0].ravel()))/self.Ws[0].ravel().shape[0]
      print 'dWs[1] diff.:',np.sum(np.abs(self.dWs[1].ravel()-emp_dWs[1].ravel()))/self.Ws[1].ravel().shape[0]
      print 'dWs[2] diff.:',np.sum(np.abs(self.dWs[2].ravel()-emp_dWs[2].ravel()))/self.Ws[2].ravel().shape[0]

      cs_copy = copy.deepcopy(self.cs)
      emp_dcs = copy.deepcopy(self.cs)
      for h in range(self.n_hidden_layers):
         for i in range(self.cs[h].shape[0]):
            self.cs[h][i] += epsilon
            output = self.use_learner(example)
            a = self.cost(output,example)[1]
            self.cs[h][i] -= epsilon
            
            self.cs[h][i] -= epsilon
            output = self.use_learner(example)
            b = self.cost(output,example)[1]
            self.cs[h][i] += epsilon
            
            emp_dcs[h][i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.cs = cs_copy
      print 'dcs[0] diff.:',np.sum(np.abs(self.dcs[0].ravel()-emp_dcs[0].ravel()))/self.cs[0].ravel().shape[0]
      print 'dcs[1] diff.:',np.sum(np.abs(self.dcs[1].ravel()-emp_dcs[1].ravel()))/self.cs[1].ravel().shape[0]
      print 'dcs[2] diff.:',np.sum(np.abs(self.dcs[2].ravel()-emp_dcs[2].ravel()))/self.cs[2].ravel().shape[0]

      U_copy = np.array(self.U)
      emp_dU = np.zeros(self.U.shape)
      for i in range(self.U.shape[0]):
         for j in range(self.U.shape[1]):
            self.U[i,j] += epsilon
            output = self.use_learner(example)
            a = self.cost(output,example)[1]
            self.U[i,j] -= epsilon

            self.U[i,j] -= epsilon
            output = self.use_learner(example)
            b = self.cost(output,example)[1]
            self.U[i,j] += epsilon

            emp_dU[i,j] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.U[:] = U_copy
      print 'dU diff.:',np.sum(np.abs(self.dU.ravel()-emp_dU.ravel()))/self.U.ravel().shape[0]

      d_copy = np.array(self.d)
      emp_dd = np.zeros(self.d.shape)
      for i in range(self.d.shape[0]):
         self.d[i] += epsilon
         output = self.use_learner(example)
         a = self.cost(output,example)[1]
         self.d[i] -= epsilon
         
         self.d[i] -= epsilon
         output = self.use_learner(example)
         b = self.cost(output,example)[1]
         self.d[i] += epsilon
         
         emp_dd[i] = (a-b)/(2.*epsilon)

      self.update_learner(example)
      self.d[:] = d_copy
      print 'dd diff.:',np.sum(np.abs(self.dd.ravel()-emp_dd.ravel()))/self.d.ravel().shape[0]


class ClusteredWeightsNNet(Learner):
   """
   Neural Network with cluster-dependent weights, for classification.

   Option ``n_stages`` is the number of training iterations.

   Options ``learning_rate`` and ``decrease_constant`` correspond
   to the learning rate and decrease constant used for stochastic
   gradient descent.

   Option ``hidden_size`` is the hidden layer size for each clustered weights.
   
   Option ``seed`` determines the seed for randomly initializing the
   weights.

   Option ``n_clusters`` is the number of clusters to extract
   with k-means. 

   Option ``n_k_means_stages`` is the number of training iterations
   for k-means. 

   Option ``n_k_means`` is the number of k-means clusterings to produce.

   Option ``n_k_means_inputs`` is the number of randomly selected
   inputs for each k-means clustering. If < 1, then will use
   all inputs.

   Option ``autoencoder_regularization`` is the weight of
   regularization, based on the denoising autoencoder objective
   (default = 0).
   
   Option ``autoencoder_missing_fraction`` is the fraction of inputs
   to mask and set to 0, for the denoising autoencoder objective
   (default = 0.1).

   Option ``activation_function`` should be string describing
   the hidden unit activation function to use. Choices are
   ``'sigmoid'`` (default), ``'tanh'`` and ``'reclin'``.
   
   **Required metadata:**

   * ``'input_size'``: Size of the input.
   * ``'targets'``: Set of possible targets.

   """

   def __init__(self, n_stages = 10, 
                learning_rate = 0.01, 
                decrease_constant = 0,
                hidden_size = 10,
                seed = 1234,
                n_clusters = 10,
                n_k_means_stages = 10,
                n_k_means = 1,
                n_k_means_inputs = -1,
                autoencoder_regularization = 0,
                autoencoder_missing_fraction = 0.1,
                activation_function = 'reclin',
                ):
       self.n_stages = n_stages
       self.stage = 0
       self.learning_rate = learning_rate
       self.decrease_constant = decrease_constant
       self.hidden_size = hidden_size
       self.seed = seed
       self.n_clusters = n_clusters
       self.n_k_means_stages = n_k_means_stages
       self.n_k_means = n_k_means
       self.n_k_means_inputs = n_k_means_inputs
       self.autoencoder_regularization = autoencoder_regularization
       self.autoencoder_missing_fraction = autoencoder_missing_fraction
       self.activation_function = activation_function
       if self.activation_function not in ['sigmoid','tanh','reclin']:
           raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

   def initialize(self,trainset):

       metadata = trainset.metadata
       self.n_classes = len(metadata['targets'])
       self.rng = np.random.mtrand.RandomState(self.seed)
       self.input_size = metadata['input_size']
       if self.n_k_means_inputs > self.input_size or self.n_k_means_inputs < 1:
           self.n_k_means_inputs = self.input_size
       
       self.Ws = []
       self.cs = []
       self.Vs = []
       self.dWs = []
       self.dcs = []
       self.dVs = []
       self.layers = []
       self.layer_acts = []
       self.dlayers = []
       self.dlayer_acts = []
       self.output_acts = []
       
       self.input = np.zeros((self.input_size,))
       self.d = np.zeros((self.n_classes,))
       self.dd = np.zeros((self.n_classes,))
       self.output = np.zeros((self.n_classes,))
       self.output_act = np.zeros((self.n_classes,))
       self.doutput_act = np.zeros((self.n_classes,))
       
       self.cluster_indices = np.zeros((self.n_k_means,),dtype='int')
              
       for k in range(self.n_k_means):
           for c in range(self.n_clusters):
               self.Ws += [(2*self.rng.rand(self.hidden_size,self.input_size)-1)/self.n_k_means_inputs]
               self.cs += [np.zeros((self.hidden_size))]
               self.Vs += [(2*self.rng.rand(self.n_classes,self.hidden_size)-1)/(self.hidden_size*self.n_k_means)]
       
               self.dWs += [np.zeros((self.hidden_size,self.input_size))]
               self.dcs += [np.zeros((self.hidden_size))]
               self.dVs += [np.zeros((self.n_classes,self.hidden_size))]
               
           self.layers += [np.zeros((self.hidden_size))]
           self.layer_acts += [np.zeros((self.hidden_size))]
       
           self.dlayers += [np.zeros((self.hidden_size))]
           self.dlayer_acts += [np.zeros((self.hidden_size))]
           self.output_acts += [np.zeros((self.n_classes,))]

       # Denoising autoencoder variables
       if self.autoencoder_regularization != 0:
           self.dae_dWs = []
           self.dae_dWsT = []

           self.input_idx = np.arange(self.input_size)
           self.dae_layers = []
           self.dae_layer_acts = []
           self.dae_dlayers = []
           self.dae_dlayer_acts = []
           self.dae_output_acts = []
           self.dae_input = np.zeros((self.input_size,))
           self.dae_d = np.zeros((self.input_size,))
           self.dae_dd = np.zeros((self.input_size,))
           self.dae_output = np.zeros((self.input_size,))
           self.dae_output_act = np.zeros((self.input_size,))
           self.dae_doutput_act = np.zeros((self.input_size,))

           for k in range(self.n_k_means):
               for c in range(self.n_clusters):
                   self.dae_dWs += [np.zeros((self.hidden_size,self.input_size))]
                   self.dae_dWsT += [np.zeros((self.input_size,self.hidden_size))]
                   
               self.dae_layers += [np.zeros((self.hidden_size))]
               self.dae_layer_acts += [np.zeros((self.hidden_size))]
               
               self.dae_dlayers += [np.zeros((self.hidden_size))]
               self.dae_dlayer_acts += [np.zeros((self.hidden_size))]
               self.dae_output_acts += [np.zeros((self.input_size,))]


       # Running k-means
       self.clusterings = []
       self.k_means_subset_inputs = []
       for k in range(self.n_k_means):
           clustering = mlfeat.k_means(n_clusters=self.n_clusters, 
                                       n_stages=self.n_k_means_stages)
           # Generate training set for k-means
           if self.n_k_means_inputs == self.input_size:
               self.k_means_subset_inputs += [None]
               def subset(ex,meta):
                   meta['input_size'] = self.n_k_means_inputs
                   return ex[0]
           else:
               subset_indices = np.arange(self.input_size)
               self.rng.shuffle(subset_indices)
               subset_indices = subset_indices[:self.n_k_means_inputs]
               self.k_means_subset_inputs += [subset_indices]
               def subset(ex,meta):
                   meta['input_size'] = self.n_k_means_inputs
                   return ex[0][subset_indices]
           k_means_trainset = mlpbgen.PreprocessedProblem(trainset,preprocess=subset)
           clustering.train(k_means_trainset)
       
           self.clusterings += [clustering]
       
       self.n_updates = 0

   def fprop(self,input):
       """
       Computes the output given some input. Puts the result in ``self.output``
       """
       self.input[:] = input
       self.output_act[:] = self.d
       for k in range(self.n_k_means):
           if self.n_k_means_inputs == self.input_size:
               c = self.clusterings[k].compute_cluster(self.input)
           else:
               c = self.clusterings[k].compute_cluster(self.input[self.k_means_subset_inputs[k]])
           idx = c + k*self.n_clusters
           self.cluster_indices[k] = c
           
           mllin.product_matrix_vector(self.Ws[idx],self.input,self.layer_acts[k])
           self.layer_acts[k] += self.cs[idx]
           #mlnonlin.sigmoid(self.layer_acts[k],self.layers[k])
           if self.activation_function == 'sigmoid':
               mlnonlin.sigmoid(self.layer_acts[k],self.layers[k])
           elif self.activation_function == 'tanh':
               mlnonlin.tanh(self.layer_acts[k],self.layers[k])
           elif self.activation_function == 'reclin':
               mlnonlin.reclin(self.layer_acts[k],self.layers[k])
           else:
               raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')
       
           mllin.product_matrix_vector(self.Vs[idx],self.layers[k],self.output_acts[k])
           self.output_act += self.output_acts[k]
       mlnonlin.softmax(self.output_act,self.output)

       if self.autoencoder_regularization != 0:
           self.dae_input[:] = input
           self.rng.shuffle(self.input_idx)
           self.dae_input[self.input_idx[:int(self.autoencoder_missing_fraction*self.input_size)]] = 0
           self.dae_output_act[:] = self.dae_d
           for k in range(self.n_k_means):
               idx = self.cluster_indices[k] + k*self.n_clusters
               
               mllin.product_matrix_vector(self.Ws[idx],self.dae_input,self.dae_layer_acts[k])
               self.dae_layer_acts[k] += self.cs[idx]
               #mlnonlin.sigmoid(self.dae_layer_acts[k],self.dae_layers[k])
               if self.activation_function == 'sigmoid':
                   mlnonlin.sigmoid(self.dae_layer_acts[k],self.dae_layers[k])
               elif self.activation_function == 'tanh':
                   mlnonlin.tanh(self.dae_layer_acts[k],self.dae_layers[k])
               elif self.activation_function == 'reclin':
                   mlnonlin.reclin(self.dae_layer_acts[k],self.dae_layers[k])
               else:
                   raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')
           
               mllin.product_matrix_vector(self.Ws[idx].T,self.dae_layers[k],self.dae_output_acts[k])
               self.dae_output_act += self.dae_output_acts[k]
           self.dae_output[:] = self.dae_output_act
          

   def bprop(self,target):
       """
       Computes the loss derivatives with respect to all parameters
       times the current learning rate.  It assumes that
       ``self.fprop(input)`` was called first. All the derivatives are
       put in their corresponding object attributes (i.e. ``self.d*``).
       """
       self.doutput_act[:] = self.output
       self.doutput_act[target] -= 1
       self.doutput_act *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
 
       self.dd[:] = self.doutput_act
       for k in range(self.n_k_means):
           c = self.cluster_indices[k]
           idx = c + k*self.n_clusters
 
           mllin.outer(self.doutput_act,self.layers[k],self.dVs[idx])
           mllin.product_matrix_vector(self.Vs[idx].T,self.doutput_act,self.dlayers[k])
           #mlnonlin.dsigmoid(self.layers[k],self.dlayers[k],self.dlayer_acts[k])
           if self.activation_function == 'sigmoid':
               mlnonlin.dsigmoid(self.layers[k],self.dlayers[k],self.dlayer_acts[k])
           elif self.activation_function == 'tanh':
               mlnonlin.dtanh(self.layers[k],self.dlayers[k],self.dlayer_acts[k])
           elif self.activation_function == 'reclin':
               mlnonlin.dreclin(self.layers[k],self.dlayers[k],self.dlayer_acts[k])
           else:
               raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

           self.dcs[idx][:] = self.dlayer_acts[k]
           mllin.outer(self.dlayer_acts[k],self.input,self.dWs[idx])

       if self.autoencoder_regularization != 0:
           self.dae_doutput_act[:] = self.dae_output
           self.dae_doutput_act[:] -= self.input
           self.dae_doutput_act *= 2*self.autoencoder_regularization*self.learning_rate/(1.+self.decrease_constant*self.n_updates)
           
           self.dae_dd[:] = self.dae_doutput_act
           for k in range(self.n_k_means):
               c = self.cluster_indices[k]
               idx = c + k*self.n_clusters
           
               mllin.outer(self.dae_doutput_act,self.dae_layers[k],self.dae_dWsT[idx])
               self.dWs[idx] += self.dae_dWsT[idx].T
               mllin.product_matrix_vector(self.Ws[idx],self.dae_doutput_act,self.dae_dlayers[k])
               #mlnonlin.dsigmoid(self.dae_layers[k],self.dae_dlayers[k],self.dae_dlayer_acts[k])
               if self.activation_function == 'sigmoid':
                   mlnonlin.dsigmoid(self.dae_layers[k],self.dae_dlayers[k],self.dae_dlayer_acts[k])     
               elif self.activation_function == 'tanh':
                   mlnonlin.dtanh(self.dae_layers[k],self.dae_dlayers[k],self.dae_dlayer_acts[k])     
               elif self.activation_function == 'reclin':
                   mlnonlin.dreclin(self.dae_layers[k],self.dae_dlayers[k],self.dae_dlayer_acts[k])     
               else:
                   raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

               self.dcs[idx] += self.dae_dlayer_acts[k]
               mllin.outer(self.dae_dlayer_acts[k],self.dae_input,self.dae_dWs[idx])
               self.dWs[idx] += self.dae_dWs[idx]               

   def update(self):
       """
       Updates the model's parameters. Assumes that ``self.fprop(input)``
       and ``self.bprop(input)`` was called first.

       It also sets all gradient information to 0.
       """
       self.d -= self.dd
       self.dd[:] = 0
       for k in range(self.n_k_means):
           c = self.cluster_indices[k]
           idx = c + k*self.n_clusters
           self.Vs[idx] -= self.dVs[idx]
           self.cs[idx] -= self.dcs[idx]
           self.Ws[idx] -= self.dWs[idx]
           self.dVs[idx][:] = 0
           self.dcs[idx][:] = 0
           self.dWs[idx][:] = 0

       if self.autoencoder_regularization != 0:          
           self.dae_d -= self.dae_dd
           self.dae_dd[:] = 0

       self.n_updates += 1
       
   def train(self,trainset):

      if self.stage == 0:
          self.initialize(trainset)
          
      for it in range(self.stage,self.n_stages):
          for input,target in trainset:
              self.fprop(input)
              self.bprop(target)
              self.update()
      self.stage = self.n_stages

   def forget(self):
       self.stage = 0

   def use(self,dataset):
       outputs = []
       for input,target in dataset:
           self.fprop(input)
           outputs += [(self.output.argmax(),np.array(self.output))]          
       return outputs
   
   def test(self,dataset):
       outputs = self.use(dataset)
       costs = np.zeros((len(dataset),1))
       t = 0
       for example,output in zip(dataset,outputs):
           costs[t,0] = example[1] != output[0]
           t+=1
       return outputs,costs

   def verify_gradients(self):
      
      print 'WARNING: calling verify_gradients reinitializes the learner'

      rng = np.random.mtrand.RandomState(1234)
      input,target = (rng.rand(20)<0.5,2)

      class fake_clustering:
          def __init__(self,cluster):
              self.cluster = cluster
          def compute_cluster(self,input):
              return self.cluster

      self.seed = 1234
      self.hidden_size = 3
      self.n_clusters = 3
      self.n_k_means_stages = 0
      self.n_k_means = 3
      self.n_k_means_inputs = 3
      self.autoencoder_regularization = 0.1
      self.autoencoder_missing_fraction = 0
      self.activation_function = 'tanh'
      self.initialize(mlpb.MLProblem([(input,target)],{'input_size':20,'targets':set([0,1,2])}))
      epsilon=1e-6
      self.learning_rate = 1
      self.decrease_constant = 0

      for l in range(10):
          input,target = (rng.rand(20)<0.5,2)
          self.clusterings = [ fake_clustering( cluster=int(rng.rand()*self.n_clusters) ) for i in range(1,self.n_k_means+1)]
          import copy
          Ws_copy = copy.deepcopy(self.Ws)
          emp_dWs = copy.deepcopy(self.Ws)
          for h in range(self.n_k_means*self.n_clusters):
            for i in range(self.Ws[h].shape[0]):
               for j in range(self.Ws[h].shape[1]):
                   self.Ws[h][i,j] += epsilon
                   self.fprop(input)
                   a = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
                   self.Ws[h][i,j] -= epsilon
                   
                   self.Ws[h][i,j] -= epsilon
                   self.fprop(input)
                   b = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
                   self.Ws[h][i,j] += epsilon
          
                   emp_dWs[h][i,j] = (a-b)/(2.*epsilon)
          
          self.bprop(target)
          self.Ws = Ws_copy
          for h in range(self.n_k_means*self.n_clusters):
              print 'dWs['+str(h)+'] diff.:',np.sum(np.abs(self.dWs[h].ravel()-emp_dWs[h].ravel()))/self.Ws[h].ravel().shape[0]
          
          cs_copy = copy.deepcopy(self.cs)
          emp_dcs = copy.deepcopy(self.cs)
          for h in range(self.n_k_means*self.n_clusters):
             for i in range(self.cs[h].shape[0]):
                self.cs[h][i] += epsilon
                self.fprop(input)
                a = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
                self.cs[h][i] -= epsilon
                
                self.cs[h][i] -= epsilon
                self.fprop(input)
                b = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
                self.cs[h][i] += epsilon
                
                emp_dcs[h][i] = (a-b)/(2.*epsilon)
          
          self.bprop(target)
          self.cs = cs_copy
          for h in range(self.n_k_means*self.n_clusters):
              print 'dcs['+str(h)+'] diff.:',np.sum(np.abs(self.dcs[h].ravel()-emp_dcs[h].ravel()))/self.cs[h].ravel().shape[0]
          
          Vs_copy = copy.deepcopy(self.Vs)
          emp_dVs = copy.deepcopy(self.Vs)
          for h in range(self.n_k_means*self.n_clusters):
            for i in range(self.Vs[h].shape[0]):
               for j in range(self.Vs[h].shape[1]):
                   self.Vs[h][i,j] += epsilon
                   self.fprop(input)
                   a = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
                   self.Vs[h][i,j] -= epsilon
                   
                   self.Vs[h][i,j] -= epsilon
                   self.fprop(input)
                   b = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
                   self.Vs[h][i,j] += epsilon
          
                   emp_dVs[h][i,j] = (a-b)/(2.*epsilon)
          
          self.bprop(target)
          self.Vs = Vs_copy
          for h in range(self.n_k_means*self.n_clusters):
              print 'dVs['+str(h)+'] diff.:',np.sum(np.abs(self.dVs[h].ravel()-emp_dVs[h].ravel()))/self.Vs[h].ravel().shape[0]
          
          d_copy = np.array(self.d)
          emp_dd = np.zeros(self.d.shape)
          for i in range(self.d.shape[0]):
             self.d[i] += epsilon
             self.fprop(input)
             a = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
             self.d[i] -= epsilon
             
             self.d[i] -= epsilon
             self.fprop(input)
             b = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
          
             self.d[i] += epsilon
             
             emp_dd[i] = (a-b)/(2.*epsilon)
          
          self.bprop(target)
          self.d[:] = d_copy
          print 'dd diff.:',np.sum(np.abs(self.dd.ravel()-emp_dd.ravel()))/self.d.ravel().shape[0]

          dae_d_copy = np.array(self.dae_d)
          emp_dae_dd = np.zeros(self.dae_d.shape)
          for i in range(self.dae_d.shape[0]):
             self.dae_d[i] += epsilon
             self.fprop(input)
             a = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
             self.dae_d[i] -= epsilon
             
             self.dae_d[i] -= epsilon
             self.fprop(input)
             b = -np.log(self.output[target]) + self.autoencoder_regularization * np.sum((self.dae_output-input)**2)
          
             self.dae_d[i] += epsilon
             
             emp_dae_dd[i] = (a-b)/(2.*epsilon)
          
          self.bprop(target)
          self.dae_d[:] = dae_d_copy
          print 'dae_dd diff.:',np.sum(np.abs(self.dae_dd.ravel()-emp_dae_dd.ravel()))/self.dae_d.ravel().shape[0]

          # Setting gradients to 0
          self.update()
