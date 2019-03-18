#include "Layer.h"


	Layer::Layer(size_t size, size_t previous) :errors(size), layer(size) {
		for (auto &neuron_obj : layer) {
			neuron_obj = new neuron(previous);
		}
	}
	Layer::Layer(size_t size) :errors(size), layer(size) {
		for (auto &inputNeuron_obj : layer) {
			inputNeuron_obj = new inputNeuron();
		}
	}
	Layer::Layer() {}
	Layer::~Layer() {
		for (auto single_neuron : layer) {
			delete single_neuron;
		}
	};
	void Layer::activateLayer(std::vector<double> &inputValues) {
		for (size_t i = 0; i < layer.size(); i++) {
			layer[i]->set_activation(inputValues[i]);
		}
	}
	void Layer::forward(Layer *previousLayer) {
		for (auto single_neuron : layer) {
			dynamic_cast<neuron*>(single_neuron)->activate(previousLayer->layer);
		}
	}
	void Layer::lastLayerDelta(vector<double> &correctActivations) {
		for (size_t i = 0; i < correctActivations.size(); i++) {
			errors[i] = layer[i]->get_activation() - correctActivations[i];
		}
	}
	inputNeuron* Layer::at(size_t i) {
		return layer[i];
	}
	vector<double> Layer::layerDelta(Layer *previousLayer) {
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

			dynamic_cast<neuron*>(layer[i])->getBias() -= error;
		}
		return backprop_errors;
	}
	void Layer::set_errors(vector<double> &_errors) {
		errors = _errors;
	}
	const vector<double>& Layer::get_errors() const {//add const
		return errors;
	}

	size_t Layer::getSize() {
		return layer.size();
	}
	vector<double> Layer::get_layer_activations() {
		vector<double> layer_activations;
		for (size_t i = 0; i < layer.size(); i++)
		{
			layer_activations.push_back(layer[i]->get_activation());

		}
		return layer_activations;
	}
	void Layer::firstInitOfLayer() {
		for (auto single_neuron : layer) {
			dynamic_cast<neuron*>(single_neuron)->initWeightsRandomly();
		}
	}

