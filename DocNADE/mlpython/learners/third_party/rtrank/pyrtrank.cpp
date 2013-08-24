#include "main.h"
#include "tuple.h"
#include "args.h"
#include "regression_tree.h"
#include "forest.h"
#include <boost/python.hpp>
#include <boost/bind.hpp>
#include <boost/thread/thread.hpp>

using namespace boost::python;

extern vector<vector<int> > fInds;

class pyrtrank_interface
{
public:
  int num_features;
  int maxdepth;
  int fk;
  int seed;
  bool par;
  vector<dt_node*> trees;
  vector< ::tuple*> training_data;

private:

  ::tuple * test_time_tuple;

public:
  
  pyrtrank_interface(int num_features, int maxdepth, int k_random_features, int seed) :
    num_features(num_features),maxdepth(maxdepth), fk(k_random_features), seed(seed), par(false),test_time_tuple(0) // not sure what it does
  {
    if(num_features<=k_random_features)
      fk = -1;

    srand(seed);
  };
  
  // destructor: free up allocated memory
  ~pyrtrank_interface() 
  { 
    if (!test_time_tuple)
      delete test_time_tuple;
    for (int i=0; i<training_data.size(); i++)
      delete training_data[i];
    for (int i=0; i<trees.size(); i++)
      delete trees[i];
  }

  void add_example(double label, double weight, boost::python::list& feat_vec)
  {
    //* +1 to the number of features, because it should contain the example ID
    //* z=1 maps missing values to UNKNOWN
    ::tuple* t = new ::tuple(num_features+1,1,0); 
    if (t == NULL)
      cout << "out of memory" << endl;

    t->features[0] = training_data.size();
    t->idx = training_data.size();
    int n =  boost::python::len(feat_vec);
    for (boost::python::ssize_t i = 0; i < n; i++) 
      t->features[i+1] = boost::python::extract<double>(feat_vec[i]);
    t->label = label;
    t->target = label;
    t->weight = weight;
    training_data.push_back(t);
  }

  void train(bool classifier, bool entropy_loss, bool bagging, bool zeros_are_missing, int n_processors)
  {
    args_t myargs;

    init_args(myargs);
    if (classifier)
      myargs.pred = ALG_MODE;
    else
      myargs.pred = ALG_MEAN;

    if (entropy_loss)
      myargs.loss = ALG_ENTROPY;
    else
      myargs.loss = ALG_SQUARED_LOSS;

    myargs.alg = ALG_BOOST; // Doesn't actually matter
    myargs.ntra = training_data.size();
    myargs.features = num_features+1;

    // This is so that UNKNOWN is mapped to a special MISSING node, and that 0 is mapped to YES,
    // like it should if the data is normalized in [0,1]
    myargs.ones = 0;//ones; 

    // If 0 mapped to MISSING is prefered
    if (zeros_are_missing)
      myargs.missing = 1;
    else
      myargs.missing = 0;

    myargs.processors = n_processors;

    if (trees.size() == 0)
      {
	add_idx(training_data);

	fInds.resize(0);
	if (myargs.processors==1)
	  presort(training_data, myargs);
	else
	  presort_p(training_data, myargs);
      }

    if(bagging)
    {
      vector< ::tuple*> sample;
      randsample(training_data,sample);
      //add_idx(sample);

      trees.push_back(new dt_node(sample,myargs,maxdepth,1,fk,par,myargs));
    }
    else
      trees.push_back(new dt_node(training_data,myargs,maxdepth,1,fk,par,myargs));
  }

//  void random_forest_mpi(const vector< ::tuple*>& train, args_t& myargs, int n_trees, vector<dt_node*>& new_trees)
//  {
//    int i;
//    for (i=0; i<n_trees; i++)
//      {
//	vector< ::tuple*> sample;
//	randsample(train,sample);
//	new_trees[i] = new dt_node(sample,myargs,maxdepth,1,fk,par,myargs);
//      }
//  }
//
//  void train_forest_mpi(bool classifier, bool entropy_loss, bool zeros_are_missing, int n_trees, int n_processors)
//  {
//    args_t myargs;
//
//    init_args(myargs);
//    if (classifier)
//      myargs.pred = ALG_MODE;
//    else
//      myargs.pred = ALG_MEAN;
//
//    if (entropy_loss)
//      myargs.loss = ALG_ENTROPY;
//    else
//      myargs.loss = ALG_SQUARED_LOSS;
//
//    myargs.alg = ALG_FOREST; // Doesn't actually matter
//    myargs.ntra = training_data.size();
//    myargs.features = num_features+1;
//
//    // This is so that UNKNOWN is mapped to a special MISSING node, and that 0 is mapped to YES,
//    // like it should if the data is normalized in [0,1]
//    myargs.ones = 0;//ones; 
//
//    // If 0 mapped to MISSING is prefered
//    if (zeros_are_missing)
//      myargs.missing = 1;
//    else
//      myargs.missing = 0;
//
//    add_idx(training_data);
//
//    fInds.resize(0);
//    if (n_processors==1)
//      presort(training_data, myargs);
//    else
//      {
//	myargs.processors = n_processors;
//	presort_p(training_data, myargs);
//	myargs.processors = 1; // Don't want to use processors for growing individual trees after
//      }
//    
//
//    // Launch jobs on several threads
//    if (n_processors > n_trees)
//      n_processors = n_trees;
//    
//    int n_trees_per_thread = n_trees / n_processors;
//    thread** threads = new thread*[n_processors];
//
//    vector< vector<dt_node*> > new_trees;
//    int i;
//    for (i=0; i < n_processors-1; i++) {
//      vector<dt_node*> thread_trees;
//      thread_trees.resize(n_trees_per_thread);
//      new_trees.push_back(thread_trees);
//    }
//    int last_thread_n_trees = n_trees - (n_processors-1)*n_trees_per_thread;
//    vector<dt_node*> thread_trees;
//    thread_trees.resize(last_thread_n_trees);
//    new_trees.push_back(thread_trees);
// 
//    fprintf(stdout, "Start threading\n");
//    for (i=0;i<n_processors-1;i++)
//      threads[i] = new thread(bind(random_forest_mpi, cref(training_data), ref(myargs), n_trees_per_thread, ref(new_trees[i]))) ;
//    threads[n_processors-1] = new thread(bind(random_forest_mpi, cref(training_data), ref(myargs), last_thread_n_trees, ref(new_trees[n_processors-1]))) ;  
//
//    for (i=0;i<n_processors;i++){
//      threads[i]->join();
//      delete threads[i];
//    }
//    fprintf(stdout, "Done threading\n");
//    delete[] threads;
//
//    // Put all new trees in the forest
//    int j;
//    for (i=0;i<n_processors-1;i++)
//      for (j=0; j<new_trees[i].size(); j++)
//	trees.push_back(new_trees[i][j]);
//  }

  void subtract_predictions_from_targets(int tree_idx, double factor)
  {
    for (int i=0;i<training_data.size(); i++)
      training_data[i]->target = training_data[i]->target - factor * dt_node::classify(training_data[i], trees[tree_idx]);
  }

  int n_trees()
  {
    return trees.size();
  }

  void delete_training_data()
  {
    for (int i=0; i<training_data.size(); i++)
      delete training_data[i];
    training_data.resize(0);
  }

  void print_features(int tree_idx)
  {
    trees[tree_idx]->print_features();
  }

  double predict(int tree_idx, boost::python::list& feat_vec)
  {
    if (!test_time_tuple)
      test_time_tuple = new ::tuple(num_features+1,1,0);

    int n = boost::python::len(feat_vec);
    for (boost::python::ssize_t i = 0; i < n ; i++) 
      test_time_tuple->features[i+1] = boost::python::extract<double>(feat_vec[i]);
    return dt_node::classify(test_time_tuple, trees[tree_idx]);
  }
};


BOOST_PYTHON_MODULE(pyrtrank)
{
  class_<pyrtrank_interface>("pyrtrank_interface", init<int, int, int, int>())
    .def("add_example", &pyrtrank_interface::add_example)
    .def("train", &pyrtrank_interface::train)
    .def("subtract_predictions_from_targets", &pyrtrank_interface::subtract_predictions_from_targets)
    .def("n_trees", &pyrtrank_interface::n_trees)
    .def("delete_training_data", &pyrtrank_interface::delete_training_data)
    .def("predict", &pyrtrank_interface::predict)
    .def("print_features", &pyrtrank_interface::print_features)
    ;
}
