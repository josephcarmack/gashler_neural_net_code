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
	cout << "Loading data...\n"; cout.flush();
	Matrix labels;	labels.loadARFF("/home/joseph/data/time_series/unemployment_rate.arff");

	Rand r(4242);
	NeuralNet* nn = new NeuralNet(r);
	vector<size_t> layers;	layers.push_back(80);	layers.push_back(30);
	nn->setTopology(layers);
	nn->init(1,1,256);
}

int main(int argc, char *argv[])
{
	enableFloatingPointExceptions();
	int ret = 1;
	try
	{
		NeuralNet::unit_test1();
		NeuralNet::unit_test2();
//		assignment11();
		ret = 0;
	}
	catch(const std::exception& e)
	{
		cerr << "An error occurred: " << e.what() << "\n";
	}
	return ret;
}
