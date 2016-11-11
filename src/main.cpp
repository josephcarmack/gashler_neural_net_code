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

void assignment11()
{
	const double pi = 3.14159265359;
	cout << "Loading data...\n"; cout.flush();
	Matrix labels; labels.loadARFF("/home/joseph/data/time_series/unemployment_rate.arff");
	Matrix feat; feat.loadARFF("/home/joseph/data/time_series/time.arff");

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
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
		NeuralNet::unit_test1();
		NeuralNet::unit_test2();
		assignment11();
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	return ret;
}
