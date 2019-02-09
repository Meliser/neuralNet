#include"neuron.h"
	default_random_engine neuron::engine(3);
	uniform_real_distribution<> neuron::distr(-0.5, 0.5);

	neuron::neuron(size_t weightVectorSize) {
		bias = 1;
		weights.reserve(weightVectorSize);
		//improve randomness
		for (size_t i = 0; i < weights.capacity(); i++)
		{
			weights.push_back(distr(engine));
			//weights.push_back(i + 1);
			//weights.push_back(0.5);
		}
	}
	neuron::~neuron()
	{
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
		//cout << "activation: " << activation << endl;
	}
	double neuron::sigmoidDerivativeZ() {
		return exp(-z) / ((1 + exp(-z))*(1 + exp(-z)));
	}
	double& neuron::getZ() {
		return z;
	}
	vector<double>& neuron::getWeights() {
		return weights;
	}
	double& neuron::getBias() {
		return bias;
	}

