// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef NEURALNET_H
#define NEURALNET_H

#include "supervised.h"
#include "vec.h"
#include <vector>

class Rand;


/// An internal class used by the NeuralNet class
class Layer
{
public:
	Matrix m_weights; // cols = in, rows = out
	Vec m_bias;
	Vec m_net;
	Vec m_activation;
	Vec m_blame;
	Matrix m_weightDelta;
	Vec m_biasDelta;

	void init(size_t in, size_t out, Rand& rand);
	void feed_forward(const Vec& in);
	void backprop(const Layer& from);
	void decay_deltas(double momentum);
	void update_deltas(const Vec& in);
	void update_weights(double learning_rate);
};




/// A multi-layer perceptron
class NeuralNet : public SupervisedLearner
{
private:
	Rand& m_rand;
	std::vector<size_t> m_topology;
	std::vector<Layer*> m_layers;
	size_t* m_pattern_indexes;

public:
	NeuralNet(Rand& r);
	virtual ~NeuralNet();

	virtual const char* name() { return "NeuralNet"; }

	/// Pass in the number of units in the hidden layers (in feed-forward order).
	/// The number of units in the output layer will be determined by the labels when you call train.
	/// This method can only be called before train is called. If it is not called, then the
	/// default topology is no hidden layers.
	void setTopology(std::vector<size_t>& topology) { m_topology = topology; }

	/// Sets all the weights to small random values
	void init(const Matrix& features, const Matrix& labels);

	/// Train the neural net
	virtual void train(const Matrix& features, const Matrix& labels);

	/// Make a prediction
	virtual void predict(const Vec& in, Vec& out);

	/// Run some unit tests
	static void unit_test1();

	void train_stochastic(const Matrix& features, const Matrix& labels, double learning_rate, double momentum);

protected:
	void feed_forward(const Vec& in);
	void present_pattern(const Vec& features, const Vec& labels);
	void compute_output_layer_blame_terms(const Vec& target);
	void backpropagate();
	void descend_gradient(double learning_rate);
};


#endif // NEURALNET_H