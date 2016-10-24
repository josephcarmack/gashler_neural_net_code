// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "filter.h"
#include "error.h"
#include "string.h"
#include "memory.h"




Filter::Filter(SupervisedLearner* pLearner, UnsupervisedLearner* pTransform, bool filterInputs)
{
	m_pLearner = pLearner;
	m_pTransform = pTransform;
	m_filterInputs = filterInputs;
	if(strcmp(pLearner->name(), "Filter") == 0)
	{
		if(((Filter*)pLearner)->m_pTransform == pTransform)
			throw Ex("Each filter requires a unique transform object");
	}
}

// virtual
Filter::~Filter()
{
	delete(m_pLearner);
	delete(m_pTransform);
}

// virtual
void Filter::filter_data(const Matrix& feat_in, const Matrix& lab_in, Matrix& feat_out, Matrix& lab_out)
{
	if(feat_in.rows() != lab_in.rows())
		throw Ex("Expected features and labels to have the same number of rows");
	if(m_filterInputs)
	{
		m_pTransform->train(feat_in);
		Matrix temp;
		temp.copyMetaData(m_pTransform->outputTemplate());
		temp.newRows(feat_in.rows());
		for(size_t i = 0; i < feat_in.rows(); i++)
			m_pTransform->transform(feat_in[i], temp[i]);
		m_pLearner->filter_data(temp, lab_in, feat_out, lab_out);
	}
	else
	{
		m_pTransform->train(lab_in);
		Matrix temp;
		temp.copyMetaData(m_pTransform->outputTemplate());
		temp.newRows(lab_in.rows());
		for(size_t i = 0; i < lab_in.rows(); i++)
			m_pTransform->transform(lab_in[i], temp[i]);
		m_pLearner->filter_data(feat_in, temp, feat_out, lab_out);
	}
}


// virtual
void Filter::train(const Matrix& features, const Matrix& labels)
{
	if(features.rows() != labels.rows())
		throw Ex("Expected features and labels to have the same number of rows");
	if(m_filterInputs)
	{
		m_pTransform->train(features);
		Matrix temp;
		temp.copyMetaData(m_pTransform->outputTemplate());
		temp.newRows(features.rows());
		for(size_t i = 0; i < features.rows(); i++)
			m_pTransform->transform(features[i], temp[i]);
		m_pLearner->train(temp, labels);
	}
	else
	{
		m_pTransform->train(labels);
		Matrix temp;
		temp.copyMetaData(m_pTransform->outputTemplate());
		temp.newRows(labels.rows());
		for(size_t i = 0; i < labels.rows(); i++)
			m_pTransform->transform(labels[i], temp[i]);
		m_pLearner->train(features, temp);
	}
}

// virtual
void Filter::predict(const Vec& in, Vec& out)
{
	if(m_filterInputs)
	{
		m_pTransform->transform(in, m_buffer);
		m_pLearner->predict(m_buffer, out);
	}
	else
	{
		m_pLearner->predict(in, m_buffer);
		m_pTransform->untransform(m_buffer, out);
	}
}



