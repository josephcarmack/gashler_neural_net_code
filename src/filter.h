// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef FILTER_H
#define FILTER_H

#include "supervised.h"
#include "unsupervised.h"
#include "vec.h"
#include <vector>



// This class wraps another supervised learner. It applies some unsupervised
// operation to the data before presenting it to the learner.
class Filter : public SupervisedLearner
{
private:
	SupervisedLearner* m_pLearner;
	UnsupervisedLearner* m_pTransform;
	bool m_filterInputs;
	Vec m_buffer;

public:
	/// This takes ownership of pLearner and pTransform.
	/// If inputs is true, then it applies the transform only to the input features.
	/// If inputs is false, then it applies the transform only to the output labels.
	/// (If you wish to transform both inputs and outputs, you must wrap a filter in a filter)
	Filter(SupervisedLearner* pLearner, UnsupervisedLearner* pTransform, bool filterInputs);

	/// Deletes the supervised learner and the transform
	virtual ~Filter();

	/// Returns the name of this learner
	virtual const char* name() { return "Filter"; }

	/// Filters the data
	virtual void filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out);

	/// Train the transform and the inner learner
	virtual void train(const Matrix& features, const Matrix& labels);

	/// Make a prediction
	virtual void predict(const Vec& in, Vec& out);

};


#endif // FILTER_H
