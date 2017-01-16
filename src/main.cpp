// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <exception>
#include <string>
#include <memory>
#include "error.h"
#include "string.h"
#include "rand.h"
#include "matrix.h"
#include "supervised.h"
#include "neuralnet.h"
#include "filter.h"
#include "nomcat.h"
#include "normalizer.h"
#include "activationFunctions.h"
#include <fstream>
#include <cmath>

using std::cout;
using std::cerr;
using std::string;
using std::vector;

void generateData()
{
	// modify mnist data set by making the images bigger
	// and offsetting the numbers by some fraction of the
	// increased size in a random fashion. Save the value
	// of the offset as a label to be predicted by a neural
	// net as part of project 13.

	Matrix mnist; 
	mnist.loadARFF("/home/joseph/data/mnist/train_feat.arff");
	
	// create new matrix for storing shifted mnist data
	size_t image_size = 28;
	size_t halo_size = 14;
	size_t new_image_size = (image_size + 2*halo_size);
	size_t num_cols = new_image_size*new_image_size;
	Matrix shifted(mnist.rows(),num_cols);

	// copy mnist data into new data set,shift, and store
	// shift in new matrix	
	Rand r(1337);
	int nx,ny;
	size_t little_index,big_index;
	Matrix shifts(mnist.rows(),2);
	for (size_t i = 0; i<mnist.rows();i++)
	{
		// generate random shift
		nx = r.next(2*halo_size+1)-halo_size;
		ny = r.next(2*halo_size+1)-halo_size;
		shifts[i][0] = nx;
		shifts[i][1] = ny;
		for (size_t row = 0; row < image_size; row++)
		{
			for (size_t col = 0; col < image_size; col++)
			{
				little_index = row*image_size + col;
				big_index = (halo_size+row-ny)*new_image_size + (halo_size+col+nx);
				shifted[i][big_index] = mnist[i][little_index];
			}
		}
	}
	shifted.saveARFF("/home/joseph/data/project14/train_shifted_mnist.arff");
	shifts.saveARFF("/home/joseph/data/project14/train_shifts.arff");
}
void assignment14()
{
	// use a MLP to predict the shifted distances from
	// the center of the picture for the shifted mnist data

	// load data
	Matrix train_feat; train_feat.loadARFF("/home/joseph/data/project14/train_shifted_mnist.arff");
	Matrix train_lab;  train_lab.loadARFF("/home/joseph/data/project14/train_shifts.arff");
	Matrix test_feat; test_feat.loadARFF("/home/joseph/data/project14/test_shifted_mnist.arff");
	Matrix test_lab;  test_lab.loadARFF("/home/joseph/data/project14/test_shifts.arff");
//	Matrix train_feat; train_feat.loadARFF("/home/joseph/data/project14/tf_subset.arff");
//	Matrix train_lab;  train_lab.loadARFF("/home/joseph/data/project14/tl_subset.arff");
//	Matrix test_feat; test_feat.loadARFF("/home/joseph/data/project14/tsf_subset.arff");
//	Matrix test_lab;  test_lab.loadARFF("/home/joseph/data/project14/tsl_subset.arff");

	// grab subset of training and test data
//	Matrix tf(1000,train_feat.cols());
//	Matrix tl(1000,train_lab.cols());
//	tf.copyBlock(0,0,train_feat,0,0,1000,train_feat.cols());
//	tl.copyBlock(0,0,train_lab,0,0,1000,train_lab.cols());
//	tf.saveARFF("/home/joseph/data/project14/tf_subset.arff");
//	tl.saveARFF("/home/joseph/data/project14/tl_subset.arff");
//	Matrix tsf(100,test_feat.cols());
//	Matrix tsl(100,test_lab.cols());
//	tsf.copyBlock(0,0,test_feat,0,0,100,test_feat.cols());
//	tsl.copyBlock(0,0,test_lab,0,0,100,test_lab.cols());
//	tsf.saveARFF("/home/joseph/data/project14/tsf_subset.arff");
//	tsl.saveARFF("/home/joseph/data/project14/tsl_subset.arff");

	// create MLP
	Rand r(1337);
	NeuralNet* nn = new NeuralNet(r);
	vector<size_t> layers;

	// one hidden layer
//	layers.push_back(268);

	// two hidden layer
//	layers.push_back(100);
//	layers.push_back(50);

	// three hidden layers
	layers.push_back(200);
	layers.push_back(60);
	layers.push_back(8);

	// create nn
	nn->setTopology(layers);
	nn->init(3136,2,train_feat.rows());

	// create filters to normalize the data
	Filter * f1 = new Filter(nn,new Normalizer(),true);
	Filter * f2 = new Filter(f1,new Normalizer(),false);

//	// train the mlp
//	f2->train(train_feat,train_lab);
//
//	// test against test data
//	double sse = f2->measureSSE(test_feat,test_lab);
//	cout << "sse = " << sse << std::endl;

	// filter normalize the data
	Matrix tf_filt;
	Matrix tl_filt;
	f2->filter_data(train_feat,train_lab,tf_filt,tl_filt);

	// train the MLP on the training data
	double tsse;
	double sse;
	double lr = 0.03;
	std::ofstream errData;
	errData.open("errorData.log");
	errData << "trSSE\ttsSSE\n";
	for (size_t i = 0; i<100; i++)
	{
		nn->train_stochastic(tf_filt,tl_filt,lr,0.0);
		sse = f2->measureSSE(test_feat,test_lab)/(double)test_feat.rows();
		tsse = f2->measureSSE(train_feat,train_lab)/(double)train_feat.rows();
		cout << "tsse=" << tsse << std::endl; cout.flush();
		lr *= 0.98;

		// log errors for this epoch
		errData << tsse << "\t" << sse << std::endl;
	}
	errData.close();
	
//	// save the predictions
	Matrix predictions(test_lab.rows(),test_lab.cols());
	Vec pred;
	cout << "making predictions...\n";
	for (size_t i=0;i<test_feat.rows();i++)
	{
		f2->predict(test_feat[i],pred);
		predictions[i].copy(pred);
		predictions[i][0] = (int) std::round(predictions[i][0]);
		predictions[i][1] = (int) std::round(predictions[i][1]);
	}
	predictions.saveARFF("/home/joseph/data/project14/predictions.arff");
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
//		NeuralNet::unit_test1();
//		NeuralNet::unit_test2();
//		generateData();
		assignment14();
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	return ret;
}
