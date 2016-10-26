// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "neuralnet.h"
#include "error.h"
#include "string.h"
#include "rand.h"
#include <math.h>
#include <cmath>

using std::vector;

//#define ACTIVATION_LOGISTIC
#define ACTIVATION_TANH


double activation(double x)
{
#if defined(ACTIVATION_LOGISTIC)
	if(x >= 700.0)
		return 1.0;
	if(x < -700.0)
		return 0.0;
	return 1.0 / (exp(-x) + 1.0);
#elif defined(ACTIVATION_TANH)
	if(x >= 700.0)
		return 1.0;
	if(x < -700.0)
		return -1.0;
	return tanh(x);
#endif
}

double activationDerivative(double net, double activation)
{
#if defined(ACTIVATION_LOGISTIC)
	return activation * (1.0 - activation);
#elif defined(ACTIVATION_TANH)
	return 1.0 - (activation * activation);
#endif
}






void Layer::init(size_t inSize, size_t outSize, Rand& rand)
{
	m_weights.setSize(outSize, inSize);
	m_weightDelta.setSize(outSize, inSize);
	m_bias.resize(outSize);
	m_biasDelta.resize(outSize);
	m_net.resize(outSize);
	m_activation.resize(outSize);
	m_blame.resize(outSize);
	double mag = std::max(0.03, 1.0 / inSize);
	for(size_t j = 0; j < outSize; j++)
	{
		for(size_t i = 0; i < inSize; i++)
			m_weights[j][i] = mag * rand.normal();
		m_net[j] = mag * rand.normal();
	}
	m_weightDelta.fill(0.0);
	m_biasDelta.fill(0.0);
}

void Layer::feed_forward(const Vec& in)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		m_net[i] = in.dotProduct(m_weights[i]) + m_bias[i];
		m_activation[i] = activation(m_net[i]);
	}
}

void Layer::backprop(const Layer& from)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		double e = 0.0;
		for(size_t j = 0; j < from.m_weights.rows(); j++)
			e += from.m_weights[j][i] * from.m_blame[j] * activationDerivative(m_net[i], m_activation[i]);
		m_blame[i] = e;
	}
}

void Layer::decay_deltas(double momentum)
{
	m_weightDelta *= momentum;
	m_biasDelta *= momentum;
}

void Layer::update_deltas(const Vec& in)
{
	for(size_t j = 0; j < m_weights.rows(); j++)
	{
		Vec& wd = m_weightDelta[j];
		for(size_t i = 0; i < m_weights.cols(); i++)
			wd[i] += in[i] * m_blame[j];
		m_biasDelta[j] += m_blame[j];
	}
}

void Layer::update_weights(double learning_rate)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		Vec& w = m_weights[i];
		Vec& wd = m_weightDelta[i];
		for(size_t j = 0; j < m_weights.cols(); j++)
			w[j] += learning_rate * wd[j];
		m_bias[i] += learning_rate * m_biasDelta[i];
	}
}








NeuralNet::NeuralNet(Rand& r)
: SupervisedLearner(), m_rand(r), m_pattern_indexes(nullptr)
{
}

// virtual
NeuralNet::~NeuralNet()
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
	delete(m_pattern_indexes);
}

void NeuralNet::init(size_t in, size_t out, size_t rows)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		delete(m_layers[i]);
	m_layers.clear();

	// create and initialize input and hidden layers
	for(size_t i = 0; i < m_topology.size(); i++)
	{
		Layer* pNewLayer = new Layer();
		pNewLayer->init(in, m_topology[i], m_rand);
		m_layers.push_back(pNewLayer);
		in = m_topology[i];
	}

	// create and initialize output layer
	Layer* pNewLayer = new Layer();
	pNewLayer->init(in, out, m_rand);
	m_layers.push_back(pNewLayer);

	// create and initialize indexing pattern array for this topology
	delete(m_pattern_indexes);
	m_pattern_indexes = new size_t[rows];
	for(size_t i = 0; i < rows; i++)
		m_pattern_indexes[i] = i;
}

// virtual
void NeuralNet::predict(const Vec& in, Vec& out)
{
	if(in.size() != m_layers[0]->m_weights.cols())
		throw Ex("input size differs from training features");
	feed_forward(in);
	out.copy(m_layers[m_layers.size() - 1]->m_activation);
}

void NeuralNet::feed_forward(const Vec& in)
{
	m_layers[0]->feed_forward(in);
	for(size_t i = 1; i < m_layers.size(); i++)
		m_layers[i]->feed_forward(m_layers[i - 1]->m_activation);
}

void NeuralNet::compute_output_layer_blame_terms(const Vec& target)
{
	Layer& output_layer = *m_layers[m_layers.size() - 1];
	for(size_t i = 0; i < target.size(); i++)
		output_layer.m_blame[i] = (target[i] - output_layer.m_activation[i]) * activationDerivative(output_layer.m_net[i], output_layer.m_activation[i]);
}

void NeuralNet::backpropagate()
{
	for(size_t i = m_layers.size() - 1; i > 0; i--)
		m_layers[i - 1]->backprop(*m_layers[i]);
}

void NeuralNet::present_pattern(const Vec& features, const Vec& labels)
{
	feed_forward(features);
	compute_output_layer_blame_terms(labels);
	backpropagate();
	m_layers[0]->update_deltas(features);
	for(size_t i = 1; i < m_layers.size(); i++)
		m_layers[i]->update_deltas(m_layers[i - 1]->m_activation);
}

void NeuralNet::descend_gradient(double learning_rate)
{
	for(size_t i = 0; i < m_layers.size(); i++)
		m_layers[i]->update_weights(learning_rate);
}

// virtual
void NeuralNet::train(const Matrix& features, const Matrix& labels)
{
	init(features.cols(), labels.cols(),features.rows());
	double learning_rate = 0.03;
	for(size_t i = 0; i < 100; i++)
	{
		train_stochastic(features, labels, 0.03, 0.0);
		learning_rate *= 0.98;
	}
}

void NeuralNet::train_stochastic(const Matrix& features, const Matrix& labels, double learning_rate, double momentum)
{
	// Shuffle the presentation order
	for(size_t j = features.rows() - 1; j > 0; j--)
		std::swap(m_pattern_indexes[j], m_pattern_indexes[m_rand.next(j)]);

	// Present the patterns for training
	for(size_t i = 0; i < features.rows(); i++)
	{
		for(size_t j = 0; j < m_layers.size(); j++)
			m_layers[j]->decay_deltas(momentum);
		size_t index = m_pattern_indexes[i];
		present_pattern(features[index], labels[index]);
		descend_gradient(learning_rate);
	}
}

void NeuralNet::update_inputs(double learning_rate,Vec& inputs)
{
	Vec negGrad(inputs.size());
	negGrad.fill(0.0);
	double blame, weight;
	// calculate the negative gradient for each input
	// then update the input by descending the gradient
	for (size_t i = 0; i<inputs.size();i++)
	{
		// loop over each unit inputs feed into
		for (size_t j = 0; j<m_topology.at(0);j++)
		{
			blame = m_layers[0]->m_blame[j];
			weight = m_layers[0]->m_weights[i][j];
			negGrad[i] += blame*weight;
		}
		// update input
		inputs[i] += negGrad[i]*learning_rate;
	}
}

void NeuralNet::train_with_images(const Matrix& X)
{
	size_t width = 64;
	size_t height = 48;
	size_t channels = X.cols()/(width*height); // should equal 3 for this problem (rgb) values
	size_t n = X.rows();
	size_t k = 2; // degrees of freedom (2 for the crane system)
	Matrix  v(n,k);
	v.fill(0.0);
	Vec features(k+2);
	Vec labels(channels); // stores rgb values for corresponding pixel
	double lr = 0.1;
	// loop variables
	size_t t,p,q,s,e,iter;
	double x,y;
	Vec pred(channels);

	std::cout << "training features with images...\n";
	for (size_t j = 0; j<10;j++)
	{
		for (size_t i = 0; i<100000;i++)
		{
			// pick a random row
			t = m_rand.next(n);
			// pick random pixel in image
			p = m_rand.next(width);
			q = m_rand.next(height);
			// create features from p,q, and v
			x = (double) p / (double) width;
			y = (double) q / (double) height;
			// build feature vector
			features[0] = x;
			features[1] = y;
			for (size_t m=0;m<k;m++)
				features[2+m]=v[t][m];
			// build label vector
			s = channels * (width*q + p);
			e = s + channels;
			iter = 0;
			for (size_t m=s;m<e;m++)
			{
				labels[iter] = X[t][m];
				iter++;
			}
			// make prediction
			predict(features,pred);
			// compute blame on layers 
			compute_output_layer_blame_terms(pred);
			backpropagate();
			// update weights and inputs (v)
			descend_gradient(lr);
			update_inputs(lr,v[t]);
		}
	}

	// save v for plotting
	std::cout << "saving results for V...\n";
	v.saveARFF("vResults.arff");
}

void NeuralNet::unit_test1()
{
	Rand rand(0);
	NeuralNet mlp(rand);
	vector<size_t> topology;
	topology.push_back(3);
	mlp.setTopology(topology);
	Matrix features(1, 2);
	Matrix labels(1, 2);
	features[0][0] = 0.3;
	features[0][1] = -0.2;
	labels[0][0] = 0.1;
	labels[0][1] = 0.0;
	mlp.init(features.cols(), labels.cols(), features.rows());

	// Set the weights
	Matrix& w1 = mlp.m_layers[0]->m_weights;
	w1[0][0] = 0.1; w1[0][1] = 0.1;
	w1[1][0] = 0.0; w1[1][1] = 0.0;
	w1[2][0] = 0.1; w1[2][1] = -0.1;
	Vec& b1 = mlp.m_layers[0]->m_bias;
	b1[0] = 0.1;
	b1[1] = 0.1;
	b1[2] = 0.0;
	Matrix& w2 = mlp.m_layers[1]->m_weights;
	w2[0][0] = 0.1; w2[0][1] = 0.1; w2[0][2] = 0.1;
	w2[1][0] = 0.1; w2[1][1] = 0.3; w2[1][2] = -0.1;
	Vec& b2 = mlp.m_layers[1]->m_bias;
	b2[0] = 0.1;
	b2[1] = -0.2;

	// Train
	mlp.train_stochastic(features, labels, 0.1, 0.0);

	// Spot check the activations
	if(std::abs(0.09966799462495 - mlp.m_layers[0]->m_activation[1]) > 1e-8)
		throw Ex("act1 wrong");
	if(std::abs(-0.16268123406035 - mlp.m_layers[1]->m_activation[1]) > 1e-8)
		throw Ex("act2 wrong");

	// Spot check the blames
	if(std::abs(0.15837584528136 - mlp.m_layers[1]->m_blame[1]) > 1e-8)
		throw Ex("blame1 wrong");
	if(std::abs(0.04457938080482 - mlp.m_layers[0]->m_blame[1]) > 1e-8)
		throw Ex("blame2 wrong");

	// Spot check the updated weights
	if(std::abs(-0.0008915876160964 - mlp.m_layers[0]->m_weights[1][1]) > 1e-8)
		throw Ex("weight1 wrong");
	if(std::abs(0.30157850028962 - mlp.m_layers[1]->m_weights[1][1]) > 1e-8)
		throw Ex("weight2 wrong");
}

