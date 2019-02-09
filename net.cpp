#include"net.h"
	double  net::learningRate = 1;
	// add constructor &
	net::net(initializer_list<int> &&topology) {
		layers.push_back(new Layer(*topology.begin()));
		for (auto it = topology.begin()+1; it !=topology.end(); it++)
		{
			layers.push_back(new Layer(*it, *(it - 1)));
		}
	}
	net::~net()
	{
		for (size_t i = 0; i < layers.size(); i++)
		{
			delete layers[i];
		}
	}
	void net::feedForward(vector<double> &inputActivations) {
		layers[0]->activate(inputActivations);
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->forward(layers[i - 1]);
		}
	}
	void net::backPropogation(vector<double> &correctActivations) {
		vector<double> previousErrors;//!!!! size
		
		layers[layers.size() - 1]->lastLayerDelta(correctActivations);
		for (size_t i = layers.size() - 1; i > 0; i--)
		{
			previousErrors = layers[i]->layerDelta(layers[i-1]);
			layers[i - 1]->set_errors(previousErrors);
		}
		//layers[layers.size() - 1]->get_errors().clear();
	}

