#include"neuron.h"
	default_random_engine neuron::engine(3);
	uniform_real_distribution<> neuron::distr(-0.5, 0.5);
	//add abstract class of neuron
	//add class of weights and bias;
	neuron::neuron(size_t weightVectorSize):weights(weightVectorSize),bias(1) {}
	neuron::~neuron(){
		cout << "~neuron()" << endl;
	}
	void neuron::activate(vector<inputNeuron*> &previous_neurons) {
		z = 0;
		for (size_t i = 0; i < weights.size(); i++)
		{
			z += previous_neurons[i]->get_activation() * weights[i];
		}
		z += bias;
		activation = 1 / (1 + exp(-z));
	}
	double neuron::sigmoidDerivativeZ() {
		double _exp = exp(-z);
		return _exp / ((1 + _exp)*(1 + _exp));
	}
	double& neuron::getZ() {
		return z;
	}
	vector<double>& neuron::getWeights() {
		return weights;
	}
	void neuron::setWeights(vector<double>& _weights)
	{
		weights = _weights;//not efficient
	}
	double& neuron::getBias() {
		return bias;
	}
	void neuron::setBias(double _bias) {
		bias = _bias;
	}
	void neuron::initWeightsRandomly()
	{
		//improve randomness
		for (auto &singleWeight:weights)
		{
			singleWeight = distr(engine);
		}
	}

