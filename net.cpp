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
		for (auto layer:layers){
			delete layer;
		}
	}
	void net::feedForward(vector<double> &inputActivations) {//template?
		layers[0]->activateLayer(inputActivations);
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->forward(layers[i - 1]);
		}
	}
	const vector<double> net::getResult() {
		return layers.back()->get_layer_activations();
	}

	vector<Layer*>& net::get_layers()
	{
		return layers;
	}

	void net::firstInit(){
		for (size_t i = 1; i < layers.size(); i++)
		{
			layers[i]->firstInitOfLayer();
		}
	}
	void saveInFile(net &obj, const char * filename)
	{
		ofstream fout(filename);
		boost::archive::text_oarchive oa(fout);
		oa << obj;
	}
	void loadFromFile(net & obj, const char * filename)
	{
		std::ifstream fin(filename);
		boost::archive::text_iarchive ia(fin);
		ia >> obj;
	}

