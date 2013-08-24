#include<iostream>
#include "math.h"

extern "C" {

  void words_list_from_counts(int *ids, int* counts, int* words, int n_counts)
  {
    int i,j,w,nw;
    for(i=0;i<n_counts;i++) {
      nw = *counts++;
      w = *ids++;
      for(j=0;j<nw;j++) {
	*words++ = w;
      }
    }  
  }
  
  void doc_mnade_tree_fprop_word_probs(double *h, double *V, double *b, int *words, 
				       bool *binary_codes, int *path_lengths, int *node_ids,
				       double * tree_probs, int n_words, int hidden_size, int tree_depth)
  {
    int i,j,k,w,l,n;
    bool c;
    double *rh, *rV, *rhflag, *rtreep;
    double p;
    bool *rbin;
    int *rnode;
    for(i=0;i<n_words;i++) {
      w = *words++;
      rbin = &binary_codes[w*tree_depth];
      l = path_lengths[w];
      rnode = &node_ids[w*tree_depth];
      rtreep = &tree_probs[i*tree_depth];
      rhflag = &h[i*hidden_size];
      for(k=0;k<l;k++) {
	n = *rnode++;
	c = *rbin++;
	rh = rhflag;
	rV = &V[n*hidden_size];
	p=b[n];
	// Compute p
	for(j=0;j<hidden_size;j++) {
	  p+= *rh++ * *rV++;
	}

	if (c)
	  *rtreep++ = 1./(1.+exp(-p));
	else
	  *rtreep++ = 1.-(1./(1.+exp(-p)));
	
      }
    }  
  }


  void doc_mnade_tree_bprop_word_probs(double *h, double *dh, double *V, double *dV, double *b, double *db, 
				       int *words, bool *binary_codes, int *path_lengths, int *node_ids, double * tree_probs, 
				       bool* to_update, int n_words, int hidden_size, int tree_depth)
  {
    int i,j,k,w,l,n;
    bool c;
    double *rh, *rdh, *rV, *rdV, *rtreep;
    double *rhflag, *rdhflag; 
    double dout;
    bool *rbin;
    int *rnode;
    for(i=0;i<n_words;i++) {
      w = *words++;
      rbin = &binary_codes[w*tree_depth];
      l = path_lengths[w];
      rnode = &node_ids[w*tree_depth];
      rtreep = &tree_probs[i*tree_depth];
      rhflag = &h[i*hidden_size];
      rdhflag = &dh[i*hidden_size];
      for(k=0;k<l;k++) {
	n = *rnode++;
	c = *rbin++;
	if (c)
	  dout = *rtreep++ - 1.;
	else
	  dout = 1. - *rtreep++;

	dout /= n_words;
	rh = rhflag;
	rdh = rdhflag;
	rV = &V[n*hidden_size];
	rdV = &dV[n*hidden_size];
	db[n] += dout;
	to_update[n] = true;
	for(j=0;j<hidden_size;j++) {
	  *rdV++ += dout * *rh++;
	  *rdh++ += dout * *rV++;
	}	
      }
    }  
  }

  void doc_mnade_tree_update_word_probs(double *V, double *dV, double *b, double *db, 
					int *words, int *path_lengths, int *node_ids,
					bool* to_update, int n_words, 
					int hidden_size, int tree_depth, double lr)
  {
    int i,j,k,w,l,n;
    double *rV, *rdV;
    int *rnode;
    for(i=0;i<n_words;i++) {
      w = *words++;
      l = path_lengths[w];
      rnode = &node_ids[w*tree_depth];
      for(k=0;k<l;k++) {
	n = *rnode++;
	if ( to_update[n] ) {
	  to_update[n] = false;
	  b[n] -= lr * db[n];
	  db[n] = 0;
	  rV = &V[n*hidden_size];
	  rdV = &dV[n*hidden_size];	  
	  for(j=0;j<hidden_size;j++) {
	    *rV++ -= lr * *rdV;
	    *rdV++ = 0;
	  }	
	}
      }
    }  
  }

  void doc_mnade_sparse_update_W(double *W, int *word_ids, double *dW, int n_words, int hidden_size, double lr)
  {
    int i,j,w;
    double *rW, *rdW;
    for(i=0;i<n_words;i++) {
      w = *word_ids++;
      rW = &W[w*hidden_size];
      rdW = &dW[w*hidden_size];
      for(j=0;j<hidden_size;j++)
      {
	*rW++ -= lr * *rdW;
	*rdW++ = 0;
      }
    }  
  }
  
}
