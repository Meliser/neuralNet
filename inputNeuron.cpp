#include"inputNeuron.h"
	double inputNeuron::get_activation() {
		return activation;
	}
	void inputNeuron::set_activation(double act) {
		activation = act;
	}
	inputNeuron::~inputNeuron()
	{
		cout << "~inputNeuron()" << endl;
	}
