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

using std::cout;
using std::cerr;
using std::string;
using std::vector;

void assignment11()
{
	const double pi = 3.14159265359;
	cout << "Loading data...\n"; cout.flush();
	Matrix labels; labels.loadARFF("/home/joseph/data/time_series/unemployment_rate.arff");
	Matrix feat; feat.loadARFF("/home/joseph/data/time_series/time.arff");

	// grap first 256 rows for training
	Matrix trainF(256,feat.cols());
	Matrix trainL(256,labels.cols());
	trainF.copyBlock(0,0,feat,0,0,256,feat.cols());
	trainL.copyBlock(0,0,labels,0,0,256,labels.cols());

	// create neural network
	Rand r(4242);
	NeuralNet* nn = new NeuralNet(r);
	vector<size_t> layers;	layers.push_back(101);
	nn->setTopology(layers);
	nn->init(1,1,256);

	// initialize weights
	for (size_t i=0;i<50;i++)
	{
		// for input layer
		nn->m_layers[0]->m_weights[i][0] = (double)(i+1)*2.0*pi;
		nn->m_layers[0]->m_weights[i+50][0] = (double)(i+1)*2.0*pi;
		nn->m_layers[0]->m_bias[i] = pi;
		nn->m_layers[0]->m_bias[i+50] = pi;
	}
	nn->m_layers[0]->m_weights[100][0] = 0.01;
	nn->m_layers[0]->m_bias[100] = 0;

	// set activation functions
	for (size_t i=0;i<100;i++)
		nn->m_layers[0]->set_activation_func(sine,cosine,i);
	nn->m_layers[0]->set_activation_func(identity,didentity,100);
	nn->m_layers[1]->set_activation_func(identity,didentity,0);

	// train nn 
	double lr = 0.03;
	for (size_t ep=0;ep<800;ep++)
	{
		nn->train_stochastic(trainF,trainL,lr,0.0);
		std::cout << "sse=" << nn->measureSSE(trainF,trainL) << std::endl;
		lr *= 0.98;
	}

	// make predictions and save to file
	Matrix predF(356,feat.cols());
	Matrix predL(356,labels.cols());
	predF.copyBlock(0,0,feat,0,0,356,feat.cols());
	predL.copyBlock(0,0,labels,0,0,356,labels.cols());
	Matrix predictions(356,1);
	predF.saveARFF("/home/joseph/data/time_series/sub_time.arff");
	predL.saveARFF("/home/joseph/data/time_series/sub_unemp.arff");
	for (size_t i=0;i<356;i++)
	{
		nn->predict(predF[i],predL[i]);
		predictions[i].copy(nn->m_layers[1]->m_activation);
	}
	predictions.saveARFF("/home/joseph/data/time_series/l1reg.arff");
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
//		NeuralNet::unit_test1();
//		NeuralNet::unit_test2();
//		assignment11();
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	return ret;
}
