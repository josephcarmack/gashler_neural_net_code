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

	Rand r(4242);
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
//	// load data for training the observation function
	cout << "Loading data...\n"; cout.flush();
	Matrix observations;	
	observations.loadARFF("/home/joseph/data/crane/observations.arff");

	// create and train observation function, generates V matrix
	Rand r(4242);
	NeuralNet* obsFunc = new NeuralNet(r);
	vector<size_t> layers;	layers.push_back(12);	layers.push_back(12);
	obsFunc->setTopology(layers);
	obsFunc->init(4,3,observations.rows()); // # feat, # labels, # data patterns
	obsFunc->train_with_images(observations);

	// load data for training the transition function
	Matrix actions;	
	actions.loadARFF("/home/joseph/data/crane/actions.arff");
	Matrix v; 
	v.loadARFF("/home/joseph/projects/gashler_neural_net_code/bin/vResults.arff");

	// build features matrix (control,state)
	Matrix rawFeat(actions.rows()-1,3);
	rawFeat.copyBlock(0,0,actions,0,0,actions.rows()-1,actions.cols());
	rawFeat.copyBlock(0,1,v,0,0,v.rows()-1,v.cols());

	// build labels matrix (next state)
	Matrix rawLabels(actions.rows()-1,2);
	rawLabels.copyBlock(0,0,v,1,0,v.rows()-1,v.cols());
	// make labels be difference between next state and current state
	for (size_t i=0;i<rawLabels.rows();i++)
	{
		rawLabels[i][0] -= v[i][0];
		rawLabels[i][1] -= v[i][1];
	}

	// create and train the transition function
	Rand rr(4242);
	NeuralNet* transFunc = new NeuralNet(rr);
	layers.clear();
	layers.push_back(6);
	transFunc->setTopology(layers);
	transFunc->init(6,2,rawFeat.rows());

	// apply normalization and nomcat filters to data
	Filter * normFeat = new Filter(transFunc,new Normalizer(),true);
	Filter * normLab  = new Filter(normFeat,new Normalizer(),false);
	Filter * nomcatFeat = new Filter(normLab,new NomCat(),true);
	Matrix filtFeatures;
	Matrix filtLabels;
	nomcatFeat->filter_data(rawFeat,rawLabels,filtFeatures,filtLabels);

	// train the transition function using stochastic gradient descent
	cout << "Training the transition function...\n";
	for(size_t i = 0; i < 10; i++)
	{
		transFunc->train_stochastic(filtFeatures, filtLabels, 0.03, 0.0);
	}

	// test crane simulator
	Matrix plan;
	plan.copyMetaData(rawFeat);
	plan.newRows(11);
	plan[0][0] = 0;
	plan[1][0] = 0;
	plan[2][0] = 0;
	plan[3][0] = 0;
	plan[4][0] = 0;
	plan[5][0] = 2;
	plan[6][0] = 2;
	plan[7][0] = 2;
	plan[8][0] = 2;
	plan[9][0] = 2;
	plan[0][1] = v[0][0];
	plan[0][2] = v[0][1];

	// run crane simulation using plan
	Vec dv;
	for (size_t i = 0; i<plan.rows()-1; i++)
	{
		nomcatFeat->predict(plan[i],dv);
		plan[i+1][1] = plan[i][1]+dv[0];
		plan[i+1][2] = plan[i][2]+dv[1];
	}

	Matrix simulation;
	simulation.copyMetaData(v);
	simulation.newRows(11);
	simulation.copyBlock(0,0,plan,0,1,plan.rows(),2);
	simulation.saveARFF("simulated.arff");

	// generate images of simulated states
	obsFunc->makeImage(simulation[0],"frame0.png");
	obsFunc->makeImage(simulation[1],"frame1.png");
	obsFunc->makeImage(simulation[2],"frame2.png");
	obsFunc->makeImage(simulation[3],"frame3.png");
	obsFunc->makeImage(simulation[4],"frame4.png");
	obsFunc->makeImage(simulation[5],"frame5.png");
	obsFunc->makeImage(simulation[6],"frame6.png");
	obsFunc->makeImage(simulation[7],"frame7.png");
	obsFunc->makeImage(simulation[8],"frame8.png");
	obsFunc->makeImage(simulation[9],"frame9.png");
	obsFunc->makeImage(simulation[10],"frame10.png");
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
