// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef UNSUPERVISED_H
#define UNSUPERVISED_H

#include "matrix.h"
#include <vector>


class Vec;

class UnsupervisedLearner
{
public:
	UnsupervisedLearner() {}
	virtual ~UnsupervisedLearner() {}

	/// Trains this unsupervised learner
	virtual void train(const Matrix& data) = 0;

	/// Returns an example of an output matrix. The meta-data of this matrix
	/// shows how output will be given. (This matrix contains no data because it has zero rows.
	/// It is only used for the meta-data.)
	virtual const Matrix& outputTemplate() = 0;

	/// Transform a single instance
	virtual void transform(const Vec& in, Vec& out) = 0;

	/// Untransform a single instance
	virtual void untransform(const Vec& in, Vec& out) = 0;
};



#endif // UNSUPERVISED_H
