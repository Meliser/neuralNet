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
	Layer(size_t size, size_t previous);
	Layer(size_t size);
	Layer();
	~Layer();
	void activateLayer(std::vector<double> &inputValues);
	void forward(Layer *previousLayer);
	void lastLayerDelta(vector<double> &correctActivations);
	inputNeuron* at(size_t i);
	vector<double> layerDelta(Layer *previousLayer);
	void set_errors(vector<double> &_errors);
	const vector<double>& get_errors() const;
	size_t getSize();
	vector<double> get_layer_activations();
	void firstInitOfLayer();

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
