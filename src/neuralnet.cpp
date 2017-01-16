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
#include <fstream>

using std::vector;






void Layer::set_activation_func(std::function<double(double)> a,
			std::function<double(double)> da,size_t unit)
{
	act.at(unit) = a;
	act_der.at(unit) = da;
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
		// initialize weights
		for(size_t i = 0; i < inSize; i++)
			m_weights[j][i] = mag * rand.normal();
		m_net[j] = mag * rand.normal();
		// set default activation functions
		std::function<double(double)> pNewActFunc = tanH;
		std::function<double(double)> pNewActFuncDer = dtanH;
		act.push_back(pNewActFunc);
		act_der.push_back(pNewActFuncDer);
	}
	m_weightDelta.fill(0.0);
	m_biasDelta.fill(0.0);
}

void Layer::debug_init()
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		for(size_t j = 0; j < m_weights.cols(); j++)
			m_weights[i][j] = 0.007 * i + 0.003 * j;
	}
	for(size_t i = 0; i < m_weights.rows(); i++)
		m_bias[i] = 0.001 * i;
}

void Layer::feed_forward(const Vec& in)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		m_net[i] = in.dotProduct(m_weights[i]) + m_bias[i];
		m_activation[i] = act[i](m_net[i]);
	}
}

void Layer::backprop(const Layer& from)
{
	for(size_t i = 0; i < m_weights.rows(); i++)
	{
		double e = 0.0;
		for(size_t j = 0; j < from.m_weights.rows(); j++)
			e += from.m_weights[j][i] * from.m_blame[j] * act_der[i](m_net[i]);
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

void Layer::l2_regularization(double lambda, double lrnRt)
{
	double decay = 1.0-lambda*lrnRt;
	for (size_t r=0;r<m_weights.rows();r++)
	{
		for (size_t c=0;c<m_weights.cols();c++)
		{
			m_weights[r][c] *= decay;
		}
//		m_bias[r] *= decay;
	}
}


void Layer::l1_regularization(double lambda, double lrnRt)
{
	double decay = lambda*lrnRt;
	for (size_t r=0;r<m_weights.rows();r++)
	{
		for (size_t c=0;c<m_weights.cols();c++)
		{
			if (std::signbit(m_weights[r][c]))
				m_weights[r][c] += decay;
			else
				m_weights[r][c] -= decay;
		}
//		if (std::signbit(m_bias[r]))
//			m_bias[r] += decay;
//		else
//			m_bias[r] -= decay;
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
//		pNewLayer->debug_init();
		m_layers.push_back(pNewLayer);
		in = m_topology[i];
	}

	// create and initialize output layer
	Layer* pNewLayer = new Layer();
	pNewLayer->init(in, out, m_rand);
//	pNewLayer->debug_init();
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
	if (output_layer.m_activation.size() != target.size())
		throw Ex("output layer size does not match label size.");
	double dif,der;
	for(size_t i = 0; i < target.size(); i++)
	{
		dif = target[i] - output_layer.m_activation[i];
		der = output_layer.act_der[i](output_layer.m_net[i]);
		output_layer.m_blame[i] = dif * der;
	}
}

void NeuralNet::backpropagate()
{
	for(int i = m_layers.size() - 1; i > 0; i--)
	{
		m_layers[i - 1]->backprop(*m_layers[i]);
	}
}

void NeuralNet::decayDeltas(double momentum)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		m_layers[i]->decay_deltas(momentum);
	}
}

void NeuralNet::regularization(double lambda,double learningRate)
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		// uncomment for l2 regularization
		m_layers[i]->l2_regularization(lambda,learningRate);
		// uncomment for l1 regularization
//		m_layers[i]->l1_regularization(lambda,learningRate);
	}
}

void NeuralNet::present_pattern(const Vec& features, const Vec& labels, double lrnRt)
{
	feed_forward(features);
	compute_output_layer_blame_terms(labels);
	backpropagate();
	// uncommment to apply regularization
	regularization(0.005,lrnRt);// lambda, learning rate
	// update deltas
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
	double sse;

	// open error log file for writing error
	std::ofstream err_log;
	err_log.open("sse.log");
	err_log << "SSE\n";
	for(size_t i = 0; i < 30; i++)
	{
		train_stochastic(features, labels, 0.03, 0.0);
		learning_rate *= 0.98;
		sse = measureSSE(features,labels);
		std::cout << "sse = " << sse << std::endl;
		err_log << sse << std::endl;
	}
	err_log.close();
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
		present_pattern(features[index], labels[index],learning_rate);
		descend_gradient(learning_rate);
	}
}

void NeuralNet::computeInputGradient(Vec& inputs, Vec& negGrad)
{
	// calculate the negative gradient for each input
	// then update the input by descending the gradient

	negGrad.resize(inputs.size());
	negGrad.fill(0.0);
	double blame, weight;

	// loop over the inputs
	for (size_t i = 0; i<inputs.size();i++)
	{
		// loop over each unit inputs feed into
		for (size_t j = 0; j<m_layers[0]->m_weights.rows();j++)
		{
			blame = m_layers[0]->m_blame[j];
			weight = m_layers[0]->m_weights[j][i];
			negGrad[i] += blame*weight;
		}
	}
}

void NeuralNet::updateInputs(double learningRate,Vec& nGrad,Vec& intrinsics,size_t startPos)
{
	// update intrinics vector
	for (size_t i=0;i<intrinsics.size();i++)
		intrinsics[i]+= nGrad[i+startPos]*learningRate;
}

void NeuralNet::train_with_images(const Matrix& X)
{
	m_width = 64;
	m_height = 48;
	size_t channels = X.cols()/(m_width*m_height); // should equal 3 for this problem (rgb) values
	size_t n = X.rows();
	size_t k = 2; // degrees of freedom (2 for the crane system)
	Matrix  v(n,k);
	std::string attr1="x",attr2="y";
	v.setAttributeName(attr1,0);
	v.setAttributeName(attr2,1);
	v.fill(0.0);
	Vec features(k+2);
	Vec labels(channels); // stores rgb values for corresponding pixel
	Vec negGrad(features.size());
	
	// loop variables
	size_t t,p,q,s,e,iter;
	double x,y;

	double lr = 0.1;
	size_t blast = 10000000;
	size_t numBlasts = 10;
	std::cout << "training features with images...\n";
	for (size_t j = 0; j<numBlasts;j++)
	{
		for (size_t i = 0; i<blast;i++)
		{
			// pick a random row
			t = m_rand.next(n);
			// pick random pixel in image
			p = m_rand.next(m_width);
			q = m_rand.next(m_height);
			// create features from p,q, and v
			x = (double) p / (double) m_width;
			y = (double) q / (double) m_height;
			// build feature vector
			features[0] = x;
			features[1] = y;
			features[2] = v[t][0];
			features[3] = v[t][1];
			// build label vector
			s = channels * (m_width*q + p);
			e = s + channels;
			iter = 0;
			for (size_t m=s;m<e;m++)
			{
				labels[iter] = (double) X[t][m]/ 255.0;
				iter++;
			}

			// present pattern
			decayDeltas(0.0);
			present_pattern(features,labels,lr);
			// update weights and inputs (v)
			computeInputGradient(features,negGrad);
			descend_gradient(lr);
			updateInputs(lr,negGrad,v[t],2);
		}
		lr *= 0.75;
		std::cout << j+1 << " blast(s) completed, decaying learning rate to " << lr << ".\n";
	}

	// save v for plotting
	std::cout << "saving results for V...\n";
	v.saveARFF("vResults.arff");
}

unsigned int NeuralNet::rgbToUint(int r, int g, int b)
{
	return 0xff000000 | ((r & 0xff) << 16) |
		((g & 0xff) << 8) | (b & 0xff);
}

void NeuralNet::makeImage(Vec& state, const char* filename)
{
	Vec in;
	in.resize(4);
	in[2] = state[0];
	in[3] = state[1];
	Vec out;
	out.resize(3);
	MyImage im;
	im.resize(m_width, m_height);
	for(size_t y = 0; y < m_height; y++)
	{
		in[1] = (double)y / m_height;
		for(size_t x = 0; x < m_width; x++)
		{
			in[0] = (double)x / m_width;
			predict(in, out);
			unsigned int color = rgbToUint(out[0] * 255, out[1] * 255, out[2] * 255);
			im.setPixel(x, y, color);
		}
	}
	im.savePng(filename);
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

	double error;
	// check the activations

	// activations for first layer
	error = std::abs(0.10955847021443 - mlp.m_layers[0]->m_activation[0]);
	if(error > 1e-10)
	{
		std::cout << "error=" << error << std::endl;
		throw Ex("act00 wrong");
	}

	error = std::abs(0.09966799462495 - mlp.m_layers[0]->m_activation[1]);
	if(error > 1e-10)
	{
		std::cout << "error=" << error << std::endl;
		throw Ex("act01 wrong");
	}

	error = std::abs(0.04995837495788 - mlp.m_layers[0]->m_activation[2]);
	if(error > 1e-10)
	{
		std::cout << "error=" << error << std::endl;
		throw Ex("act02 wrong");
	}

	// activations for second layer
	error = std::abs(0.12525717909304 - mlp.m_layers[1]->m_activation[0]);
	if(error > 1e-10)
	{
		std::cout << "error=" << error << std::endl;
		throw Ex("act10 wrong");
	}

	error = std::abs(-0.16268123406035 - mlp.m_layers[1]->m_activation[1]);
	if(error > 1e-10)
	{
		std::cout << "error=" << error << std::endl;
		throw Ex("act11 wrong");
	}

	// Spot check the blames
	if(std::abs(0.15837584528136 - mlp.m_layers[1]->m_blame[1]) > 1e-10)
		throw Ex("blame1 wrong");
	if(std::abs(0.04457938080482 - mlp.m_layers[0]->m_blame[1]) > 1e-10)
		throw Ex("blame2 wrong");

	// Spot check the updated weights
	if(std::abs(-0.0008915876160964 - mlp.m_layers[0]->m_weights[1][1]) > 1e-10)
		throw Ex("weight1 wrong");
	if(std::abs(0.30157850028962 - mlp.m_layers[1]->m_weights[1][1]) > 1e-10)
		throw Ex("weight2 wrong");

	std::cout << "passed unit test 1...\n";
}

void NeuralNet::unit_test2()
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

	// feed forward labels then do backpropagation 
	mlp.feed_forward(features[0]); 
	mlp.compute_output_layer_blame_terms(labels[0]);
	mlp.backpropagate();

	// calc the input updates
	Vec inGrad(features.cols());
	inGrad.fill(0.0);
	mlp.computeInputGradient(features[0],inGrad);
	mlp.updateInputs(1.0,inGrad,features[0],0);

	// Spot check the updated inputs 
	double error = std::abs(0.29949132921729 - features[0][0]);
	if(error > 1e-10)
	{
		std::cout <<"Error="<< error << std::endl;
		throw Ex("feature[0][0] update is wrong");
	}
	error = std::abs(-0.19685308226483 - features[0][1]);
	if(error > 1e-10)
	{
		std::cout <<"Error="<< error << std::endl;
		throw Ex("feature[0][1] update is wrong");
	}
	std::cout << "passed unit test 2...\n";
}

// debug methods
void NeuralNet::printWeights()
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		std::cout << "----------------\nlayer["<<i<<"]:\n\n";
		std::cout << "weights:\n\n";
		m_layers[i]->m_weights.print(std::cout);
		std::cout << std::endl;
		std::cout << std::endl;
		std::cout << "bias:\n\n";
		m_layers[i]->m_bias.print(std::cout);
		std::cout << std::endl;
		std::cout << std::endl;
	}
}

void NeuralNet::printBlame()
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		std::cout << "----------------\nlayer["<<i<<"]:\n\n";
		std::cout << "blame = ";
		m_layers[i]->m_blame.print(std::cout);
		std::cout << std::endl;
	}
}

void NeuralNet::printActivations()
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		std::cout << "----------------\nlayer["<<i<<"]:\n\n";
		std::cout << "activation = ";
		m_layers[i]->m_activation.print(std::cout);
		std::cout << std::endl;
	}
}

void NeuralNet::printNets()
{
	for(size_t i = 0; i < m_layers.size(); i++)
	{
		std::cout << "----------------\nlayer["<<i<<"]:\n\n";
		std::cout << "net = ";
		m_layers[i]->m_net.print(std::cout);
		std::cout << std::endl;
	}
}
