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
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle and Stanislas Lauly ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle and Stanislas Lauly OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle and Stanislas Lauly.

import sys, os

sys.argv.pop(0);	# Remove first argument

# Check if all options are provided
if 8 != len(sys.argv):
    print "Usage: python trainDocNADE.py max_iter hidden_size learning_rate activation_function libsvm_trainset_file libsvm_valid_file libsvm_test_file voc_size"
    sys.exit()

import numpy as np
import fcntl, copy
from mlpython.learners.topic_modeling import DocNADE
import mlpython.misc.io as mlio
import mlpython.mlproblems.generic as mlpb

# Helper function to load LIBSVM data
def load(file_path,input_size=13649,load_to_memory=True):
    """
    Loads LIBSVM dataset in path ``file_path``.
    """
    dataFile = os.path.expanduser(file_path)
    
    def load_line(line):
        return mlio.libsvm_load_line(line,
                                     convert_target=str,
                                     sparse=True,
                                     input_size=input_size,
                                     input_type=np.int32)[0]

    # Get data
    data = mlio.load_from_file(dataFile,load_line)
    if load_to_memory:
        data = [x for x in data]
        length = len(data)
    else:
        length = 0
        stream = open(dataFile)
        for l in stream:
            length+=1
        stream.close()
            
    # Get metadata
    data_meta = {'input_size':input_size,'length':length}
    
    return (data,data_meta)

# Get script options
max_iter = int(sys.argv[0])
hidden_size = int(sys.argv[1])
learning_rate = float(sys.argv[2])
activation_function = sys.argv[3]

# Path to data sets
trainSetPath = sys.argv[4]
validSetPath = sys.argv[5]
testSetPath = sys.argv[6]
inputSize = int(sys.argv[7])

# Create DocNADE learner object
docNadeObject = DocNADE(n_stages=1, 
                         hidden_size=hidden_size, 
                         learning_rate=learning_rate, 
                         activation_function=activation_function)

# Load the data
train_data,train_metadata = load(trainSetPath,inputSize)
valid_data,valid_metadata = load(validSetPath,inputSize)
test_data,test_metadata = load(testSetPath,inputSize)

# Create MLProblems (data structure used for data sets in MLPython)
trainset = mlpb.MLProblem(train_data,train_metadata)
validset = mlpb.MLProblem(valid_data,valid_metadata)
testset = mlpb.MLProblem(test_data,test_metadata)

# Training wiht early-stopping
best_val_error = np.inf
best_it = 0
look_ahead = 10
n_incr_error = 0  # Nb. of consecutive increase in error
for stage in range(1,max_iter+1,1):
    if not n_incr_error < look_ahead:
        break
    docNadeObject.n_stages = stage # Ask for one more training iteration
    docNadeObject.train(trainset)  # Train some more
    n_incr_error += 1

    print 'Evaluating on validation set'
    outputs, costs = docNadeObject.test(validset)
    error = np.mean(costs,axis=0)[0]

    print 'Error: ' + str(error)
    if error < best_val_error:
        # Update best error (NLL), iteration and model
        best_val_error = error
        best_it = stage
        n_incr_error = 0
        best_model = copy.deepcopy(docNadeObject)


#Saving best_model
dirPath = 'models_' + os.path.split(trainSetPath)[1] + '/' 
if not os.path.exists(dirPath):
    os.makedirs(dirPath)
stringOption = "_" + "_".join([str(max_iter), str(hidden_size), str(learning_rate), activation_function])   
mlio.save(best_model,dirPath + "modelDocNADE" + stringOption)

# Preparing result line
## Compute results

## Get outputs and costs for each example
outputs_tr,costs_tr = best_model.test(trainset)
outputs_v,costs_v = best_model.test(validset)
outputs_t,costs_t = best_model.test(testset)

## Compute NLL
train_nll = np.mean(costs_tr,axis=0)[0]
valid_nll = np.mean(costs_v,axis=0)[0]
test_nll = np.mean(costs_t,axis=0)[0]

## Prepare result line to append to result file
line = "\t".join([str(max_iter),str(hidden_size),str(learning_rate),activation_function])
line += "\t" + "\t".join([str(best_it),str(train_nll),str(valid_nll),str(test_nll)]) + "\n"

# Preparing result file
results_file = 'results_DocNADE_file.txt'
if not os.path.exists(results_file):
    # Create result file if doesn't exist
    header_line = ""
    header_line += "\t".join(['max_iter','hidden_size','learning_rate','activation_function',
                          'best_it','train_nll','valid_nll','test_nll']) + '\n'
    f = open(results_file, 'w')
    f.write(header_line)
    f.close()

f = open(results_file, "a")
fcntl.flock(f.fileno(), fcntl.LOCK_EX)
f.write(line)
f.close() # unlocks the file

