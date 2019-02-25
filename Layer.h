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
	void activateLayer(std::vector<double> &inputValues) {
		for (size_t i = 0; i < layer.size(); i++)
		{
			layer[i]->set_activation(inputValues[i]);
		}
	}
	void forward(Layer *previousLayer) {
		for (size_t i = 0; i < layer.size(); i++)
		{
			dynamic_cast<neuron*>(layer[i])->activate(previousLayer->layer);
		}
	}
	void lastLayerDelta(vector<double> &correctActivations) {
		for (size_t i = 0; i < correctActivations.size(); i++)
		{
			errors[i] = layer[i]->get_activation() - correctActivations[i];
		}
	}
	vector<double> layerDelta(Layer *previousLayer) {
		double error;
		vector<double> backprop_errors(previousLayer->getSize());
		for (size_t i = 0; i < layer.size(); i++)
		{
			error = 2 * errors[i] * dynamic_cast<neuron*>(layer[i])->sigmoidDerivativeZ();
			for (size_t j = 0; j < previousLayer->getSize(); j++)
			{
				// don't count for first layer
				backprop_errors[j] += dynamic_cast<neuron*>(layer[i])->getWeights()[j] * error;//!!!
				dynamic_cast<neuron*>(layer[i])->getWeights()[j] -= error * previousLayer->getByIndex(j)->get_activation();
			}

			dynamic_cast<neuron*>(layer[i])->getBias() -=  error;
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
	vector<double> get_layer_activations() {
		vector<double> layer_activations;
		for (size_t i = 0; i < layer.size(); i++)
		{
			layer_activations.push_back(layer[i]->get_activation());

		}
		return layer_activations;
	}
private:
	std::vector<inputNeuron*> layer;
	std::vector<double> errors;
};
