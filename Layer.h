#pragma once
#include<vector>
#include "inputNeuron.h"
#include "neuron.h"
#include<boost/archive/text_oarchive.hpp>
#include<boost/archive/text_iarchive.hpp>
using namespace std;
class Layer
{
public:
	Layer(size_t size,size_t previous):errors(size),layer(size) {
		for (auto &neuron_obj:layer){
			neuron_obj = new neuron(previous);
		}
	}
	Layer(size_t size):errors(size),layer(size) {
		for (auto &inputNeuron_obj : layer){
			inputNeuron_obj = new inputNeuron();
		}
	}
	Layer(){}
	~Layer() {
		for (auto single_neuron : layer) {
			delete single_neuron;
		}
	};
	void activateLayer(std::vector<double> &inputValues) {
		for (size_t i = 0; i < layer.size(); i++){
			layer[i]->set_activation(inputValues[i]);
		}
	}
	void forward(Layer *previousLayer) {
		for (auto single_neuron : layer) {
			dynamic_cast<neuron*>(single_neuron)->activate(previousLayer->layer);
		}
	}
	void lastLayerDelta(vector<double> &correctActivations) {
		for (size_t i = 0; i < correctActivations.size(); i++){
			errors[i] = layer[i]->get_activation() - correctActivations[i];
		}
	}
	inputNeuron* at(size_t i) {
		return layer[i];
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
				dynamic_cast<neuron*>(layer[i])->getWeights()[j] -= error * previousLayer->at(j)->get_activation();
			}
			
			dynamic_cast<neuron*>(layer[i])->getBias() -=  error;
		}
		return backprop_errors;
	}
	void set_errors(vector<double> &_errors) {
		errors = _errors;
	}
	vector<double>& get_errors() {//add const
		return errors;
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
	void firstInitOfLayer() {
		for (auto single_neuron : layer) {
			dynamic_cast<neuron*>(single_neuron)->initWeightsRandomly();
		}
	}

private:
	std::vector<inputNeuron*> layer;
	std::vector<double> errors;
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & errors;
		ar.register_type(static_cast<neuron*>(NULL));
		ar & layer;
	}
};
