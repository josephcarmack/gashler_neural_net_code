// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef NEURALNET_H
#define NEURALNET_H

#include "supervised.h"
#include "vec.h"
#include <vector>
#include <functional>
#include "image.h"
#include "activationFunctions.h"

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
	std::function<double(double)> act,act_der;

	void init(size_t in, size_t out, Rand& rand);
	void debug_init();
	void feed_forward(const Vec& in);
	void backprop(const Layer& from);
	void decay_deltas(double momentum);
	void update_deltas(const Vec& in);
	void update_weights(double learning_rate);
	void set_activation_func(std::function<double(double)> a,
			std::function<double(double)> da);
};




/// A multi-layer perceptron
class NeuralNet : public SupervisedLearner
{
public:
	Rand& m_rand;
	std::vector<Layer*> m_layers;
private:
	std::vector<size_t> m_topology;
	size_t* m_pattern_indexes;

	// image member vars
	size_t m_width; size_t m_height;

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
	void init(size_t in, size_t out, size_t rows);

	/// Train the neural net
	virtual void train(const Matrix& features, const Matrix& labels);

	/// Make a prediction
	virtual void predict(const Vec& in, Vec& out);

	/// Run some unit tests
	static void unit_test1();
	static void unit_test2();

	void train_stochastic(const Matrix& features, const Matrix& labels, double learning_rate, double momentum);

	void train_with_images(const Matrix& X);

	/// methods for printing images
	unsigned int rgbToUint(int r, int g, int b);
	void makeImage(Vec& state, const char* filename);

	// debug methods
	void printWeights();
	void printBlame();
	void printActivations();
	void printNets();

protected:
	void feed_forward(const Vec& in);
	void present_pattern(const Vec& features, const Vec& labels);
	void compute_output_layer_blame_terms(const Vec& target);
	void backpropagate();
	void descend_gradient(double learning_rate);
	void computeInputGradient(Vec& inputs, Vec& negGrad);
	void updateInputs(double learning_rate,Vec& nGrad,Vec& intrinsics,size_t startPos);
	void decayDeltas(double momentum);
};


#endif // NEURALNET_H
