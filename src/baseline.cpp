#include "baseline.h"

BaselineLearner::BaselineLearner()
{
}

// virtual
BaselineLearner::~BaselineLearner()
{
}

// virtual
void BaselineLearner::train(const Matrix& features, const Matrix& labels)
{
	mode.resize(labels.cols());
	for(size_t i = 0; i < labels.cols(); i++)
	{
		if(labels.valueCount(i) == 0)
			mode[i] = labels.columnMean(i);
		else
			mode[i] = labels.mostCommonValue(i);
	}
}

// virtual
void BaselineLearner::predict(const Vec& in, Vec& out)
{
	out.copy(mode);
}
