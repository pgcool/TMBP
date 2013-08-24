HOW TO INSTALL

- These scripts rely on a version of the MLPython library, which is provided in the zip archive. To use it, you need to do the following:

   1. Append to your PYTHONPATH environment variable the parent path to the mlpython subdirectory

   2. Append to your CPATH environment variable the path to the NumPy C header files. 
      To figure out where they are, you can import NumPy in python:

      >>> import numpy
      >>> numpy
      <module 'numpy' from '/Library/Frameworks/EPD64.framework/Versions/6.3/lib/python2.6/site-packages/numpy/__init__.pyc'>

      which will reveal the path of the directory where NumPy is installed 
      (in this exemple, /Library/Frameworks/EPD64.framework/Versions/6.3/lib/python2.6/site-packages/numpy/). 

   3. run make in the mlpython subdirectory

  For more information go to:

    http://www.dmi.usherb.ca/~larocheh/mlpython/install.html#install


HOW TO USE

- The script trainDocNADE.py will train the model DocNADE and save it in a repository. It will also save all the information with the negative log likelyhood in a results folder. Here's how to use the script:

    python trainDocNADE.py max_iter hidden_size learning_rate activation_function libsvm_trainset_file libsvm_valid_file libsvm_test_file voc_size

    - Option ``max_iter`` is the maximum number of training iterations.
    
    - Option ``hidden_size`` should be a positive integer specifying
      the number of hidden units (features).
    
    - Options ``learning_rate`` is the learning rate (default=0.001).
    
    - Option ``activation_function`` should be string describing
      the hidden unit activation function to use. Choices are
      ``'sigmoid'`` (default), ``'tanh'`` and ``'reclin'``.
    
    - Option ``libsvm_trainset_file``,  ``libsvm_valid_file`` and ``libsvm_test_file``
      are the full path to the training, the validation and the testing sets. They
      must be in libsvm format.
    
    - Option ``voc_size`` is the size of the vocabulary.
    

- The script extractRepresentations.py will print to the standard output the extracted representations (hidden units) of the documents in a given data set (in libsvm format).

    python extractRepresentation.py model_file libsvm_dataset_file

    - Option ``model_file`` is a model trained and saved using the script trainDocNADE.py.
    
    - Option ``libsvm_dataset_file`` is the data that you want to convert into
      the representation extracted by the model.

