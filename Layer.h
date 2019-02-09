#pragma once
#include<vector>
#include "inputNeuron.h"
#include "neuron.h"
using namespace std;
class Layer
{
public:
	Layer(size_t size,size_t previous):errors(size) {
		for (size_t i = 0; i < size; i++)
		{
			layer.push_back(new neuron(previous));
		}
		//errors.reserve(size);
	}
	Layer(size_t size):errors(size) {
		for (size_t i = 0; i < size; i++)
		{
			layer.push_back(new inputNeuron());
		}
	}
	~Layer() {
		for (size_t i = 0; i < layer.size(); i++)
		{
			delete layer[i];
		}
	};
	void activate(std::vector<double> &inputValues) {
		for (size_t i = 0; i < layer.size(); i++)
		{
			layer[i]->set_activation(inputValues[i]);
		}
	}
	void forward(Layer *previousLayer) {
		for (size_t j = 0; j < layer.size(); j++)
		{
			dynamic_cast<neuron*>(layer[j])->activate(previousLayer->layer);
			
		}
	}
	void lastLayerDelta(vector<double> &correctActivations) {
		for (size_t i = 0; i < correctActivations.size(); i++)
		{
			//errors.push_back((layer[i]->get_activation() - correctActivations[i]));
			errors[i] = layer[i]->get_activation() - correctActivations[i];
		}
	}
	vector<double> layerDelta(Layer *previousLayer) {
		double error;
		vector<double> backprop_errors(previousLayer->getSize());
		for (size_t j = 0; j < layer.size(); j++)
		{
			error = 2 * errors[j] * dynamic_cast<neuron*>(layer[j])->sigmoidDerivativeZ();
			for (size_t k = 0; k < previousLayer->getSize(); k++)
			{
				// don't count for first layer
				backprop_errors[k] += dynamic_cast<neuron*>(layer[j])->getWeights()[k] * error;//!!!
				dynamic_cast<neuron*>(layer[j])->getWeights()[k] -= error * previousLayer->getByIndex(k)->get_activation();//in loop
			}

			dynamic_cast<neuron*>(layer[j])->getBias() -=  error;
		}
		return backprop_errors;
	}
	void set_errors(vector<double> _errors) {
		errors = _errors;
	}
	vector<double>& get_errors() {
		return errors;
	}
	inputNeuron* getByIndex(size_t i) {
		return layer[i];
	}
	size_t getSize() {
		return layer.size();
	}
private:
	std::vector<inputNeuron*> layer;
	std::vector<double> errors;
};
