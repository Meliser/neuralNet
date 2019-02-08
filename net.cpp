#include"net.h"
	double  net::learningRate = 1;
	// add constructor &
	net::net(initializer_list<int> &&topology) {
	
		//auto it = topology.begin();
		//vector<inputNeuron*> layer;
		
		//input_layer = new inputLayer(*topology.begin());
		layers.push_back(new Layer(*topology.begin()));
		for (auto it = topology.begin()+1; it !=topology.end(); it++)
		{
			layers.push_back(new Layer(*it, *(it - 1)));
		}
		/*for (size_t i = 0; i < topology.size(); i++)
		{
			for (size_t j = 0; j < *it; j++)
			{
				if (i == 0) {
					layer.push_back(new inputNeuron());
				}
				else
				{
					layer.push_back(new neuron(*(it-1)));
				}
			}
			it++;
			layers.push_back(layer);
			layer.clear();
		}*/
	}
	net::~net()
	{
		//delete input_layer;
		for (size_t i = 0; i < layers.size(); i++)
		{
			delete layers[i];
		}
		/*for (size_t i = 0; i < layers.size(); i++)
		{
			for (size_t j = 0; j < layers[i].size(); j++)
			{
				delete layers[i][j];
			}
		}*/
	}
	void net::feedForward(vector<double> &inputActivations) {
		/*for (size_t i = 0; i < layers[0].size(); i++)
		{
			layers[0][i]->set_activation(inputActivations[i]);
		}*/
		layers[0]->activate(inputActivations);
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->forward(layers[i - 1]);
			//for (size_t j = 0; j < layers[i]->getSize(); j++)
			//{
			//	dynamic_cast<neuron*>(layers[i][j])->activate(layers[i - 1]);
			//	// for debug
			//	if (i == layers.size() - 1) {
			//		cout << "Output activation: " << layers[i][j]->get_activation() << endl;
			//	}
			//}
		}
	}
	void net::backPropogation(vector<double> &correctActivations) {
		double error;
		vector<double> previousErrors;//!!!! size
		// debug
		//cout << "cost function: " << (layers[layers.size() - 1][0]->get_activation() - correctActivations[0])*(layers[layers.size() - 1][0]->get_activation() - correctActivations[0]) << endl;

		layers[layers.size() - 1]->lastLayerDelta(correctActivations);
		for (size_t i = layers.size() - 1; i > 0; i--)
		{
			previousErrors = layers[i]->layerDelta(layers[i-1]);//!!!!
			layers[i - 1]->set_errors(previousErrors);
			//vector<double> errors(layers[i - 1].size());
			/*if (i == layers.size() - 1) {
				for (size_t j = 0; j < correctActivations.size(); j++)
				{
					previousErrors.push_back((layers[i][j]->get_activation() - correctActivations[j]));
				}
			}*/
			//for (size_t j = 0; j < layers[i].size(); j++)
			//{
			//	error = 2 * previousErrors[j] * dynamic_cast<neuron*>(layers[i][j])->sigmoidDerivativeZ();
			//	for (size_t k = 0; k < layers[i - 1].size(); k++)
			//	{
			//		// don't count for first layer
			//		errors[k] += dynamic_cast<neuron*>(layers[i][j])->getWeights()[k] * error;
			//		dynamic_cast<neuron*>(layers[i][j])->getWeights()[k] -= learningRate * error * layers[i - 1][k]->get_activation();//in loop
			//	}

			//	dynamic_cast<neuron*>(layers[i][j])->getBias() -= learningRate * error;
			//}
			//previousErrors = errors;
			//previousErrors.shrink_to_fit();
		}
		layers[layers.size() - 1]->get_errors().clear();
	}

