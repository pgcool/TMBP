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
The package ``learners.third_party.rtrank`` contains modules using the
RT-Rank library to learn ensembles of decision trees.  These modules
only invoke the implementation of decision tree learning within
RT-Rank. To make it accessible within Python, a Python extension
must be compiled using Boost.Python, Boost.Thread and Boost.Jam.

To RT-Rank through MLPython, do the following:

1. download RT-Rank 1.5 from here: https://sites.google.com/site/rtranking/download
2. copy the content of subdirectory cart/ into this package directory
3. create a Jamroot file by doing ``cp Jamroot.template Jamroot`` in the package directory
   and edit Jamroot by changing PUT_PATH_TO_BOOST_HERE by the path to your copy 
   of Boost (e.g. /home/USER/software/boost_1_46_0)
4. compile Python extension simply by running the command ``bjam release`` in package directory 
   (requires Boost.Python, Boost.Thread and Boost.Jam)
5  add path to the package directory to your LD_LIBRARY_PATH environment variable for
   linux, or DYLD_LIBRARY_PATH for Mac OS X

And that should do it! Try 'import pyrtrank' to see if your installation is working.

Currently, ``learner.third_party.rtrank`` contains the following modules:

* ``learning.third_party.rtrank.regression``:    Ensemble of regression trees, based on the RT-Rank library.

"""
