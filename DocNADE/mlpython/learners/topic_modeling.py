# Copyright 2012 Hugo Larochelle, Stanislas Lauly. All rights reserved.
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
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle, Stanislas Lauly ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle, Stanislas Lauly OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle, Stanislas Lauly.

"""
The ``learners.topic_modeling`` module contains Learners meant for topic modeling
problems. The MLProblems for these Learners should be iterators over inputs,
which correspond to sparse representations of words counts:
they are pairs ``(freq,words)`` where ``freq`` is the vector
of frequences for each word present in the document, and ``words`` is the vector
of IDs for those words.

The currently implemented algorithms are:

* TopicModel:             the interface for all TopicModel objects that learn representation for documents.
* ReplicatedSoftmax:      the Replicated Softmax undirected topic model.
* DocNADE:                the Document Neural Autoregressive Distribution Estimator (DocNADE).
* InformationRetrieval:   applieds a given TopicModel to an information retrieval task.
"""

from generic import Learner
import numpy as np
import mlpython.mlproblems.generic as mlpb
import mlpython.mathutils.nonlinear as mlnonlin
import mlpython.mathutils.linalg as mllin
import mlpython.mathutils.topic_modeling as tm_utils
import mlpython.datasets.nips_russ as nr
import scipy.sparse as sparse
import copy
import itertools

class TopicModel(Learner):
    """
    Interface for all TopicModel objects that learn representation for documents.

    The only additional requirement from Learner is to define a method
    ``compute_document_representation(example)`` that outputs the representation
    for some given example (a ``(freq,words)`` pair).
    """

    def compute_document_representation(self,example):
        """
        Return the document representation of some given example.
        """

        raise NotImplementedError("Subclass should have implemented this method")


class ReplicatedSoftmax(TopicModel):
    """
    Replicated Softmax undirected topic model.
    
    Option ``n_stages`` is the number of training iterations.
    
    Options ``learning_rate`` and ``decrease_constant`` correspond
    to the learning rate and decrease constant used for stochastic
    gradient descent.
    
    Option ``hidden_size`` should be a positive integer specifying
    the number of hidden units (features).
    
    Option ``k_contrastive_divergence_steps`` is the number of
    Gibbs sampling iterations used by contrastive divergence.
    
    Option ``mean_field`` indicates that mean-field inference
    should be used to generate the negative statistics for learning.
    
    Option ``seed`` determines the seed for randomly initializing the
    weights.
    
    **Required metadata:**
    
    * ``'input_size'``:  Vocabulary size
    
    | **Reference:** 
    | Replicated Softmax: an Undirected Topic Model
    | Salakhutdinov and Hinton
    | http://www.utstat.toronto.edu/~rsalakhu/papers/repsoft.pdf
    
    """
    
    def __init__(self, n_stages, 
                 learning_rate = 0.01, 
                 decrease_constant = 0,
                 hidden_size = 100,
                 k_contrastive_divergence_steps = 1,
                 mean_field = False,
                 seed = 1234
                 ):
        self.n_stages = n_stages
        self.stage = 0
        self.learning_rate = learning_rate
        self.decrease_constant = decrease_constant
        self.k_contrastive_divergence_steps = k_contrastive_divergence_steps
        self.hidden_size = hidden_size
        self.mean_field = mean_field
        self.seed = seed
        
    def train(self,trainset):
        if self.stage == 0:
            self.initialize(trainset)
        for it in range(self.stage,self.n_stages):
            for example in trainset:
                self.update_learner(example)
        self.stage = self.n_stages

    def forget(self):
        self.stage = 0

    def use(self,dataset):
        outputs = []
        for example in dataset:
            outputs += [self.use_learner(example)]
        return outputs
            
    def test(self,dataset):
        outputs = self.use(dataset)
        costs = []
        for example,output in itertools.izip(dataset,outputs):
            costs += [self.cost(output,example)]

        return outputs,costs

    def initialize(self,trainset):
        self.rng = np.random.mtrand.RandomState(self.seed)
        self.input_size = trainset.metadata['input_size']
        if self.hidden_size <= 0:
            raise ValueError('hidden_size should be > 0')
        
        self.W = (2*self.rng.rand(self.hidden_size,self.input_size)-1)/self.input_size
        self.c = np.zeros((self.hidden_size))
        
        self.b = np.ones((self.input_size))*0.01
        n = 0
        words = np.zeros((self.input_size))
        for words_sparse in trainset:
            words[:] = 0
            words[words_sparse[1]] = words_sparse[0]
            self.b += words
            n += np.sum(words)
        self.b = np.log(self.b)
        self.b -= np.log(n)
        
        self.deltaW = np.zeros((self.hidden_size,self.input_size))
        self.deltac = np.zeros((self.hidden_size))
        self.deltab = np.zeros((self.input_size))
        
        self.input = np.zeros((self.input_size))
        self.hidden = np.zeros((self.hidden_size))
        self.hidden_act = np.zeros((self.hidden_size))
        self.hidden_prob = np.zeros((self.hidden_size))
        
        self.neg_input = np.zeros((self.input_size))
        self.neg_input_act = np.zeros((self.input_size))
        self.neg_input_prob = np.zeros((self.input_size))
        self.neg_hidden_act = np.zeros((self.hidden_size))
        self.neg_hidden_prob = np.zeros((self.hidden_size))
        
        self.neg_stats = np.zeros((self.hidden_size,self.input_size))
        
        self.n_updates = 0

    def update_learner(self,example):
        self.input[:] = 0
        self.input[example[1]] = example[0]
        n_words = int(self.input.sum())
        
        # Performing CD-k
        mllin.product_matrix_vector(self.W,self.input,self.hidden_act)
        self.hidden_act += self.c*n_words
        mlnonlin.sigmoid(self.hidden_act,self.hidden_prob)
        self.neg_hidden_prob[:] = self.hidden_prob
        
        for k in range(self.k_contrastive_divergence_steps):
            if self.mean_field:
               self.hidden[:] = self.neg_hidden_prob
            else: 
               np.less(self.rng.rand(self.hidden_size),self.neg_hidden_prob,self.hidden)
        
            mllin.product_matrix_vector(self.W.T,self.hidden,self.neg_input_act)
            self.neg_input_act += self.b
            mlnonlin.softmax(self.neg_input_act,self.neg_input_prob)
            if self.mean_field:
               self.neg_input[:] = n_words*self.neg_input_prob
            else:
               self.neg_input[:] = self.rng.multinomial(n_words,self.neg_input_prob)
        
            mllin.product_matrix_vector(self.W,self.neg_input,self.neg_hidden_act)
            self.neg_hidden_act += self.c*n_words
            mlnonlin.sigmoid(self.neg_hidden_act,self.neg_hidden_prob)
        
        mllin.outer(self.hidden_prob,self.input,self.deltaW)
        mllin.outer(self.neg_hidden_prob,self.neg_input,self.neg_stats)
        self.deltaW -= self.neg_stats
        
        np.subtract(self.input,self.neg_input,self.deltab)
        np.subtract(self.hidden_prob,self.neg_hidden_prob,self.deltac)
        
        self.deltaW *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
        self.deltab *= self.learning_rate/(1.+self.decrease_constant*self.n_updates)
        self.deltac *= n_words*self.learning_rate/(1.+self.decrease_constant*self.n_updates)         
        
        self.W += self.deltaW
        self.b += self.deltab
        self.c += self.deltac
        
        self.n_updates += 1

    def use_learner(self,example):
        self.input[:] = 0
        self.input[example[1]] = example[0]
        output = np.zeros((self.hidden_size))
        mllin.product_matrix_vector(self.W,self.input,self.hidden_act)
        self.hidden_act += self.c*self.input.sum()
        mlnonlin.sigmoid(self.hidden_act,output)
        
        return [output]

    def compute_document_representation(self,word_counts_sparse):
        self.input[:] = 0
        self.input[word_counts_sparse[1]] = word_counts_sparse[0]
        output = np.zeros((self.hidden_size,))
        mllin.product_matrix_vector(self.W,self.input,self.hidden_act)
        self.hidden_act += self.c*self.input.sum()
        mlnonlin.sigmoid(self.hidden_act,output)
        return output
 
    def cost(self,outputs,example):
        hidden = outputs[0]
        self.input[:] = 0
        self.input[example[1]] = example[0]
        mllin.product_matrix_vector(self.W.T,hidden,self.neg_input_act)
        self.neg_input_act += self.b
        mlnonlin.softmax(self.neg_input_act,self.neg_input_prob)
        
        return [ np.sum((self.input-self.input.sum()*self.neg_input_prob)**2) ]
 
    def verify_learning(self):
        print 'WARNING: calling verify_sum_to_one reinitializes the learner'        
        self.hidden_size = 6
        input_size = 10
        self.learning_rate = 0.01
        words = np.zeros((10,))
        words[3] = 1
        words[7] = 1
        trainset = mlpb.MLProblem([words],
                                  {'input_size':input_size})
        self.initialize(trainset)
        self.b[:] = 0
        words_neg = np.zeros((10,))
        for t in range(1,10001):
            p = np.exp(np.dot(words,self.b)+np.sum(np.log(1+np.exp(np.dot(self.W,words)+words.sum()*self.c))))
            s = 0
            words_neg[:] = 0
            for i in range(input_size):
                for j in range(input_size):
                    words_neg[:] = 0
                    words_neg[i] += 1
                    words_neg[j] += 1
                    s += np.exp(np.dot(words_neg,self.b)+np.sum(np.log(1+np.exp(np.dot(self.W,words_neg)+words_neg.sum()*self.c))))
            print p/s
            self.n_stages = t
            self.train(trainset)


class DocNADE(TopicModel):
    """
    Neural autoregressive distribution estimator for topic model.
    
    Option ``n_stages`` is the number of training iterations.
    
    Option ``hidden_size`` should be a positive integer specifying
    the number of hidden units (features).
    
    Options ``learning_rate`` is the learning rate (default=0.001).
    
    Option ``activation_function`` should be string describing
    the hidden unit activation function to use. Choices are
    ``'sigmoid'`` (default), ``'tanh'`` and ``'reclin'``.
    
    Option ``normalize_by_document_size`` normalize the learning by
    the size of the documents.
    
    Option ``hidden_bias_scaled_by_document_size`` scale the bias
    of the hidden units by the document size.
    
    Option ``seed`` determines the seed for randomly initializing the
    weights.
    
    **Required metadata:**
    
    * ``'input_size'``:  Vocabulary size
    
    | **Reference:** 
    | A Neural Autoregressive Topic Model
    | Larochelle and Lauly
    | http://www.dmi.usherb.ca/~larocheh/publications/docnade.pdf
    """

    def __init__(self, n_stages=10,
                 hidden_size = 100,
                 learning_rate = 0.001,
                 activation_function = 'sigmoid',
                 testing_ensemble_size = 1,
                 normalize_by_document_size = True,
                 hidden_bias_scaled_by_document_size = False,
                 seed = 1234
                 ):
        self.n_stages = n_stages
        self.stage = 0
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.hidden_size = hidden_size
        self.testing_ensemble_size = testing_ensemble_size
        self.normalize_by_document_size = normalize_by_document_size
        self.hidden_bias_scaled_by_document_size = hidden_bias_scaled_by_document_size
        self.seed = seed
        
    def initialize(self,trainset):
        self.stage = 0
        self.rng = np.random.mtrand.RandomState(self.seed)
        self.voc_size = trainset.metadata['input_size']

	# Create word tree
        def get_binary_codes_rec(beg,end,depth,binary_codes,node_ids,path_lengths):
            if end-beg == 1:
                path_lengths[beg] = depth
                return

            i,j,k = beg,beg + np.floor((end-beg)/2),end
            binary_codes[i:j,depth] = True
            binary_codes[j:k,depth] = False
            node_ids[i:k,depth] = j
            get_binary_codes_rec(i,j,depth+1,binary_codes,node_ids,path_lengths)
            get_binary_codes_rec(j,k,depth+1,binary_codes,node_ids,path_lengths)

        def get_binary_codes(voc_size):
            tree_depth = np.int32(np.ceil(np.log2(voc_size)))
            binary_codes = np.zeros((voc_size,tree_depth),dtype=bool)
            node_ids = np.zeros((voc_size,tree_depth),dtype=np.int32)
            path_lengths = np.zeros((voc_size),dtype=np.int32)
            get_binary_codes_rec(0,voc_size,0,binary_codes,node_ids,path_lengths)
            return binary_codes,path_lengths,node_ids

        self.binary_codes,self.path_lengths,self.node_ids = get_binary_codes(self.voc_size)
        self.tree_depth = self.binary_codes.shape[1]

        self.W = self.rng.rand(self.voc_size,self.hidden_size)/(self.voc_size*self.hidden_size)
        self.V = self.rng.rand(self.voc_size,self.hidden_size)/(self.voc_size*self.hidden_size)
        #self.V = np.zeros((self.voc_size,self.hidden_size))
        self.c = np.zeros((self.hidden_size))

        # Initialize b to base rate
        self.freq_pos = np.ones((self.voc_size))*0.01
        self.freq_neg = np.ones((self.voc_size))*0.01
        cnt = 0
        for word_counts, word_ids in trainset:
            cnt+=1
            if cnt >50000:
                break
            for i in range(self.tree_depth):
                # Node at level i
                for k in range(len(word_ids)):
                    n = self.node_ids[word_ids[k],i]
                    c = self.binary_codes[word_ids[k],i]
                    if c:
                        # Nb. of left decisions
                        self.freq_pos[n] += word_counts[k]
                    else:
                        # Nb. of right decisions
                        self.freq_neg[n] += word_counts[k]

        p = self.freq_pos / (self.freq_pos + self.freq_neg)
        # Convert marginal probabilities to sigmoid bias
        self.b = -np.log(1/p-1)
        
        self.dV = np.zeros((self.voc_size,self.hidden_size))
        self.dW = np.zeros((self.voc_size,self.hidden_size))
        self.db = np.zeros((self.voc_size))
        self.dc = np.zeros((self.hidden_size))
        
        self.to_update = np.zeros((self.voc_size,),dtype=bool)

    def compute_document_representation(self,word_counts_sparse):
        new_inp = np.zeros((self.voc_size,))
        new_inp[word_counts_sparse[1]] = word_counts_sparse[0]
        out =  np.zeros((1,self.hidden_size))
        self.apply_activation((self.c + np.dot(new_inp,self.W)).reshape((1,-1)),out)
        return out.reshape((-1,))

    def apply_activation(self, input_data, output):
	"""
        Apply the activation function
        """
        if self.activation_function == 'sigmoid':
            mlnonlin.sigmoid(input_data,output)
        elif self.activation_function == 'tanh':
            mlnonlin.tanh(input_data,output)
        elif self.activation_function == 'reclin':
            mlnonlin.reclin(input_data,output)
        elif self.activation_function == 'softmax':
            m = input_data.max(axis=1)
            output[:] = np.exp(input_data-m.reshape((-1,1)))
            output[:] /= output.sum(axis=1).reshape((-1,1))
        else:
            raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

    def apply_dactivation(self, output, doutput, dinput):
        """
        Apply the derivative of the activatiun fonction
	"""
        if self.activation_function == 'sigmoid':
            mlnonlin.dsigmoid(output,doutput,dinput)
        elif self.activation_function == 'tanh':
            mlnonlin.dtanh(output,doutput,dinput)
        elif self.activation_function == 'reclin':
            mlnonlin.dreclin(output,doutput,dinput)
        elif self.activation_function == 'softmax':
            dinput[:] = output*(doutput-(doutput*output).sum(axis=1).reshape((-1,1)))
	else:
            raise ValueError('activation_function must be either \'sigmoid\', \'tanh\' or \'reclin\'')

    def fprop_word_probs(self):
        """
        Computes the tree decision probs for all p(w_i | w_i<).
        """
        self.probs = np.ones((self.h.shape[0],self.tree_depth))
        tm_utils.doc_mnade_tree_fprop_word_probs(self.h,self.V,self.b,self.words,self.binary_codes,self.path_lengths,
                                                 self.node_ids,self.probs)
        
    def bprop_word_probs(self):
        """
        Computes db, dV and dh
        """
        self.dh = np.zeros(self.h.shape)
        tm_utils.doc_mnade_tree_bprop_word_probs(self.h,self.dh,self.V,self.dV,self.b,self.db,self.words,self.binary_codes,
                                                 self.path_lengths,self.node_ids,self.probs,self.to_update)

    def update_word_probs(self):
        """
        Updates b and V
        """
        if self.normalize_by_document_size:
            tm_utils.doc_mnade_tree_update_word_probs(self.V,self.dV,self.b,self.db,self.words,
                                                      self.path_lengths,self.node_ids,self.to_update,self.learning_rate)
        else:
            tm_utils.doc_mnade_tree_update_word_probs(self.V,self.dV,self.b,self.db,self.words,
                                                      self.path_lengths,self.node_ids,self.to_update,self.learning_rate*len(self.words))

    def fprop(self,word_ids,word_counts):
        self.word_ids = word_ids
        self.words = tm_utils.words_list_from_counts(word_ids,word_counts)
        self.rng.shuffle(self.words)
        self.act = np.zeros((len(self.words),self.hidden_size))
        np.add.accumulate(self.W[self.words[:-1],:],axis=0,out=self.act[1:,:])
        if self.hidden_bias_scaled_by_document_size:
           self.act += self.c*len(self.words)
        else:
           self.act += self.c
        self.h = np.zeros((len(self.words),self.hidden_size))
        self.apply_activation(self.act,self.h)
        self.fprop_word_probs()

    def bprop(self):
        self.bprop_word_probs()
        self.dact = np.zeros(self.act.shape)
        self.apply_dactivation(self.h, self.dh, self.dact)
        if self.hidden_bias_scaled_by_document_size:
           self.dc[:] = self.dact.sum(axis=0)*len(self.words)
        else:
           self.dc[:] = self.dact.sum(axis=0)
        dacc_input = np.zeros((len(self.words),self.hidden_size))
        np.add.accumulate(self.dact[:0:-1,:],axis=0,out=dacc_input[-2::-1,:])
        mllin.multiple_row_accumulate(dacc_input,self.words,self.dW)

    def update(self):
        self.update_word_probs()
        if self.normalize_by_document_size:
            self.c -= self.learning_rate * self.dc
            tm_utils.doc_mnade_sparse_update_W(self.W,self.word_ids,self.dW,self.learning_rate)
        else:
            self.c -= self.learning_rate * len(self.words) * self.dc
            tm_utils.doc_mnade_sparse_update_W(self.W,self.word_ids,self.dW,self.learning_rate*len(self.words))
            

    def train(self, trainset):
        
        while self.stage < self.n_stages:
            if self.stage == 0:
                self.initialize(trainset)
            self.stage += 1

            for word_counts, word_ids in trainset:
                self.fprop(word_ids,word_counts)
                self.bprop()
                self.update()

    def test(self, testset):
        costs = []
        outputs = self.use(testset)
        for output in outputs:
            costs += [[-np.mean(output)]]
        return outputs,costs

    def use(self,testset):
        rng_test_time = np.random.mtrand.RandomState(1234)
        tmp_rng = self.rng
        self.rng = rng_test_time

        outputs = []
        for word_counts, word_ids in testset:
            o = 0
            for i in range(self.testing_ensemble_size):
                self.fprop(word_ids,word_counts)
                o += np.log(self.probs).sum(axis=1)
            outputs += [o/self.testing_ensemble_size]
        self.rng = tmp_rng
        return outputs

    def verify_sums_to_one(self):
        print 'WARNING: calling verify_sum_to_one reinitializes the learner'        
        self.hidden_size = 6
        input_size = 10
        self.learning_rate = 1
        epsilon=1e-6
        word_ids = np.array([0,3,7,8,9])
        word_counts = np.array([3,2,1,7,20])
        trainset = mlpb.MLProblem([(word_counts,word_ids)],
                                  {'input_size':input_size})
        self.initialize(trainset)
                
        log_sum_p = -np.inf
        sum_p = 0
        for i in range(input_size):
            for j in range(input_size):
                for k in range(input_size):
                    self.words = np.array([i,j,k],dtype="int32")
                    self.act = np.zeros((len(self.words),self.hidden_size))
                    np.add.accumulate(self.W[self.words[:-1],:],axis=0,out=self.act[1:,:])
                    if self.hidden_bias_scaled_by_document_size:
                       self.act += self.c*len(self.words)
                    else:
                       self.act += self.c
                    self.h = np.zeros((len(self.words),self.hidden_size))
                    self.apply_activation(self.act,self.h)
                    self.fprop_word_probs()
                    log_p_i = np.log(self.probs).sum()
                    m = max(log_p_i,log_sum_p)
                    log_sum_p = m + np.log(np.exp(log_p_i-m)+np.exp(log_sum_p-m))
                    sum_p += np.exp(log_p_i)

        print "Sums to",np.exp(log_sum_p)#,sum_p

    def verify_gradients(self):
        print 'WARNING: calling verify_gradients reinitializes the learner'

        self.activation_function = "softmax"
        self.hidden_size = 6
        input_size = 10
        self.learning_rate = 0.01
        epsilon=1e-6
        word_ids = np.array([0,3,7,8,9])
        word_counts = np.array([3,2,1,7,20])
        trainset = mlpb.MLProblem([(word_counts,word_ids)],
                                  {'input_size':input_size})
        self.initialize(trainset)
        self.learning_rate = 1
        self.c = self.rng.rand(self.hidden_size)

        # Compute all derivatives
        rng_test_time = np.random.mtrand.RandomState(1234)
        tmp_rng = self.rng
        self.rng = rng_test_time
        self.fprop(word_ids,word_counts)
        self.bprop()
        self.rng = tmp_rng

        # Estimate derivatives by finite differences
        W_copy = np.array(self.W)
        lim_dW = np.zeros(self.W.shape)
        for i in range(self.W.shape[0]):
            for j in range(self.W.shape[1]):
                self.W[i,j] += epsilon
                outputs,costs = self.test(trainset)
                a = costs[0][0]
                
                self.W[i,j] -= 2.*epsilon

                outputs,costs = self.test(trainset)
                b = costs[0][0]
                self.W[i,j] += epsilon

                lim_dW[i,j] = (a-b)/(2.*epsilon)
                
        print 'dW diff.:',np.sum(np.abs(self.dW.ravel()-lim_dW.ravel()))/self.W.ravel().shape[0]

        b_copy = np.array(self.b)
        lim_db = np.zeros(self.b.shape)
        for i in range(self.b.shape[0]):
            self.b[i] += epsilon
            outputs,costs = self.test(trainset)
            a = costs[0][0]
            
            self.b[i] -= 2.*epsilon

            outputs,costs = self.test(trainset)
            b = costs[0][0]
            self.b[i] += epsilon
            
            lim_db[i] = (a-b)/(2.*epsilon)
        
        print 'db diff.:',np.sum(np.abs(self.db.ravel()-lim_db.ravel()))/self.b.ravel().shape[0]
        
        V_copy = np.array(self.V)
        lim_dV = np.zeros(self.V.shape)
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                self.V[i,j] += epsilon
                outputs,costs = self.test(trainset)
                a = costs[0][0]
                
                self.V[i,j] -= 2.*epsilon

                outputs,costs = self.test(trainset)
                b = costs[0][0]
                self.V[i,j] += epsilon

                lim_dV[i,j] = (a-b)/(2.*epsilon)
                
        print 'dV diff.:',np.sum(np.abs(self.dV.ravel()-lim_dV.ravel()))/self.V.ravel().shape[0]

        c_copy = np.array(self.c)
        lim_dh = np.zeros(self.c.shape)
        for i in range(self.c.shape[0]):
            self.c[i] += epsilon
            outputs,costs = self.test(trainset)
            a = costs[0][0]
            
            self.c[i] -= 2.*epsilon

            outputs,costs = self.test(trainset)
            b = costs[0][0]
            self.c[i] += epsilon
            
            lim_dh[i] = (a-b)/(2.*epsilon)
        
        print 'dc diff.:',np.sum(np.abs(self.dc.ravel()-lim_dh.ravel()))/self.c.ravel().shape[0]



class InformationRetrieval(Learner):
    """
    Information retrieval based on a topic model.
    
    Option ``learnerObj`` is an instance of a (trained) topic model.
    
    Option ``list_percRetrieval`` is a list of retrieval percentages at which to evaluate the precision.
    
    **Required metadata:**
    
    * ``'target_size'``:  Number of classes (for data with multiple labels)
    
    
    """

    def __init__(self,
                 learnerObj,
                 list_percRetrieval = [0.64, 0.256, 1]
                 ):
        self.learnerObj = learnerObj
        list_percRetrieval.sort() # make sure that it's in increasing order
        self.list_percRetrieval = np.array(list_percRetrieval)

    def preProLearnerObj(self, example):
        inp,target = example
        out = self.learnerObj.compute_document_representation(inp)
        return (self.unitVec(out),target)

    def convertData(self, dataset):
        """
        Convert the data in a higher representation (hidden units) learned by learnerObj. Return
        a matrix of inputs (each row is a example) and a matrix of targets. For problem's with
        one label, the matrix targets have just one column.
        """
        targetExample = dataset.peak()[1]
        fOneLabel = isinstance(targetExample,int)

        if fOneLabel:
            mat_train_targets = np.zeros((len(dataset), 1))
        else:
            mat_train_targets = np.zeros((len(dataset), dataset.metadata['target_size']))
        
        
        if self.learnerObj is None:
            raise ValueError('learnerObj must be set')
        else:
           mat_train_inputs = np.zeros((len(dataset), self.learnerObj.hidden_size))
           for i, example in enumerate(dataset):
              exPrePro = self.preProLearnerObj(example)
              mat_train_inputs[i,:] = exPrePro[0]
              mat_train_targets[i,:] = exPrePro[1]
                                    
        return [mat_train_inputs, mat_train_targets]

    def unitVec(self,vec):
        norm = np.linalg.norm(vec)
        return vec/norm

    def angleDistance(self,a,b):
        return np.dot(a,b)
    
    def train(self, dataset):
        """
        Convert and store the database.
        """
        self.mat_database_input, self.mat_database_target = self.convertData(dataset)
        self.nb_data = len(self.mat_database_input)
        self.list_totalRetrievalCount = np.floor(self.nb_data*self.list_percRetrieval)

    def use(self, queryset):
        """
        Compute the closest example (the first retrieval percentages of retrieval percentages) 
        and give there indexes in output.
        """
        mat_query_input, mat_query_target = self.convertData(queryset)

        targetExample = queryset.peak()[1]
        fOneLabel = isinstance(targetExample,int)
        outputs = []
        import itertools
        for query in itertools.izip(mat_query_input,mat_query_target):
            outputs += [self.closestExample(query)]
            
        return outputs

    def test(self, queryset):
        """
        Compute the closest example (the first retrieval percentages of retrieval percentages) 
        and give there indexes in outputs for each query.
        
        Compute the percentage of good prediction for each label of each query in costs.
        """
        mat_query_input, mat_query_target = self.convertData(queryset)

        targetExample = queryset.peak()[1]
        fOneLabel = isinstance(targetExample,int)
        costs = []
        outputs = []
        import itertools
        for query in itertools.izip(mat_query_input,mat_query_target):
            if (fOneLabel):
                cost, output = self.precisionGoodLabel(query)
            else:
                cost, output = self.precisionMultiLabel(query)

            costs += [cost]
            outputs += [output]

        return outputs, costs

    def closestExample(self, query):
        nb_data = len(self.mat_database_input)
        list_totalRetrievalCount = np.floor(nb_data*self.list_percRetrieval)
        vec_similarity = -np.dot(self.mat_database_input, query[0])
        vec_simIndexSorted = np.argsort(vec_similarity)

        return vec_simIndexSorted[:self.list_totalRetrievalCount[0]]
        
    def precisionGoodLabel(self,query):
        """
        For Data with one label per example. Return percentage of good prediction (list_percPrecision)
        for each retrieval and the index of the closest example for the first retrieval.
        """
        vec_similarity = -np.dot(self.mat_database_input, query[0])
        vec_simIndexSorted = np.argsort(vec_similarity)
        classQuery = query[1] # Which class does the query belong to
        list_percPrecision = np.zeros(len(self.list_percRetrieval))
        
        vec_goodLabel = self.mat_database_target[vec_simIndexSorted,0] == classQuery
        countGoodLabel = 0
        for indexRetrieval,totalRetrievalCount in enumerate(self.list_totalRetrievalCount):
            if indexRetrieval==0:
                countGoodLabel += np.sum(vec_goodLabel[:totalRetrievalCount])
            else:
                countGoodLabel += np.sum(vec_goodLabel[lastTotalRetrievalCount:totalRetrievalCount])
            
            list_percPrecision[indexRetrieval] = countGoodLabel/float(totalRetrievalCount)
            lastTotalRetrievalCount = totalRetrievalCount
              
        return list_percPrecision, vec_simIndexSorted[:self.list_totalRetrievalCount[0]]

    def precisionMultiLabel(self,query):
        """
        For Data with multiple label per example. Return percentage of good prediction (list_percPrecision)
        for each retrieval and the index of the closest example for the first retrieval.
        """
        vec_similarity = -np.dot(self.mat_database_input, query[0])
        vec_simIndexSorted = np.argsort(vec_similarity)
        list_indexClassQuery = [index for index,val in enumerate(query[1]) if val==1] # Which class does the query belong to
        mat_allCurve = np.zeros((len(list_indexClassQuery), len(self.list_percRetrieval)))
        indexMatAllCurve = 0
        for indexClassQuery in list_indexClassQuery:
           vec_goodLabel = self.mat_database_target[vec_simIndexSorted, indexClassQuery]
           countGoodLabel = 0
           for indexRetrieval,totalRetrievalCount in enumerate(self.list_totalRetrievalCount):
              if indexRetrieval==0:
                 countGoodLabel += np.sum(vec_goodLabel[:totalRetrievalCount])
              else:
                 countGoodLabel += np.sum(vec_goodLabel[lastTotalRetrievalCount:totalRetrievalCount])
              
              mat_allCurve[indexMatAllCurve,indexRetrieval] = countGoodLabel/float(totalRetrievalCount)
              lastTotalRetrievalCount = totalRetrievalCount
           indexMatAllCurve += 1   
        
        return mat_allCurve, vec_simIndexSorted[:self.list_totalRetrievalCount[0]]
