#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "supervised.h"
#include "error.h"
#include "string.h"
#include "rand.h"
#include "vec.h"

using std::vector;

// virtual
void SupervisedLearner::filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out)
{
	feat_out.copy(feat_in);
	lab_out.copy(lab_in);
}

size_t SupervisedLearner::countMisclassifications(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Mismatching number of rows");
	Vec pred;
	size_t mis = 0;
	for(size_t i = 0; i < features.rows(); i++)
	{
		predict(features[i], pred);
		const Vec& lab = labels[i];
		for(size_t j = 0; j < lab.size(); j++)
		{
			if(pred[j] != lab[j])
			{
				mis++;
			}
		}
	}
	return mis;
}

// virtual
void SupervisedLearner::trainIncremental(const Vec& feat, const Vec& lab)
{
	throw Ex("Sorry, this learner does not support incremental training");
}
