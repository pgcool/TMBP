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

# Check if every option(s) from parent's script are here.
if 2 != len(sys.argv):
    print "Usage: python extractRepresentation.py model_file libsvm_dataset_file"
    sys.exit()

import mlpython.learners.topic_modeling
from mlpython.learners.topic_modeling import DocNADE
import mlpython.misc.io as mlio
import numpy as np
from subprocess import check_output

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


best_model = mlio.load(sys.argv[0])
print best_model.__class__
if best_model.__class__ == mlpython.learners.topic_modeling.DocNADE:
    inputSize = best_model.voc_size
else:
    print "Model object is not supported"
    sys.exit()

# Print the higher representation (hidden units) for each example.
data,metadata = load(sys.argv[1],inputSize)
for input in data:
    print " ".join([ "%.8f"%i for i in best_model.compute_document_representation(input)])
