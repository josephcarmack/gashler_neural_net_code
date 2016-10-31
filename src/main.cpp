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

using std::cout;
using std::cerr;
using std::string;
using std::vector;

void do_mnist()
{
	cout << "Loading data...\n"; cout.flush();
	Matrix train_feat;	train_feat.loadARFF("/home/joseph/data/mnist/train_feat.arff");
	Matrix train_lab;	train_lab.loadARFF("/home/joseph/data/mnist/train_lab.arff");
	Matrix test_feat;	test_feat.loadARFF("/home/joseph/data/mnist/test_feat.arff");
	Matrix test_lab;	test_lab.loadARFF("/home/joseph/data/mnist/test_lab.arff");

	Rand r(1234);
	NeuralNet* nn = new NeuralNet(r);
	vector<size_t> layers;	layers.push_back(80);	layers.push_back(30);
	nn->setTopology(layers);

	Filter* f1 = new Filter(nn, new Normalizer(), true);
	Filter f2(f1, new NomCat(), false);

	Matrix feat_filtered;
	Matrix lab_filtered;
	f2.filter_data(train_feat, train_lab, feat_filtered, lab_filtered);

	nn->init(feat_filtered.cols(), lab_filtered.cols(),train_feat.rows());
	for(size_t i = 0; i < 10; i++)
	{
		cout << "Testing...\n"; cout.flush();
		size_t mis = f2.countMisclassifications(test_feat, test_lab);
		cout << "Misclassifications: " << to_str(mis) << "\n";
		cout << "Training...\n";
		cout.flush();
		nn->train_stochastic(feat_filtered, lab_filtered, 0.03, 0.0);
	}
}

void do_assignment9()
{
	cout << "Loading data...\n"; cout.flush();
	Matrix observations;	
	Matrix actions;	
	Matrix setIn(1,4);
	Matrix setOut(1,3);
	observations.loadARFF("/home/joseph/data/crane/observations.arff");
	actions.loadARFF("/home/joseph/data/crane/actions.arff");

	Rand r(1234);
	NeuralNet* nn = new NeuralNet(r);
	Filter* f1 = new Filter(nn, new Normalizer(), false);

	Matrix normed_obser;
	f1->filter_data(observations,observations,normed_obser,normed_obser);

	vector<size_t> layers;	layers.push_back(12);	layers.push_back(12);
	nn->setTopology(layers);
	nn->init(4,3,observations.rows()); // # feat, # labels, # data patterns
	nn->train_with_images(normed_obser);
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
		NeuralNet::unit_test1();
		NeuralNet::unit_test2();
//		do_mnist();
		do_assignment9();
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	return ret;
}
