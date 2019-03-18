#include "Trainer.h"


Trainer::Trainer(net &obj,size_t _epochs):netToTrain(obj),epochs(_epochs)
{

}

Trainer::~Trainer()
{
	delete region;
	delete region2;
}
double shrinky(double x, double in_min = 0, double in_max = 255, double out_min = 0, double out_max = 1)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}
void Trainer::backPropogation()
{
		vector<double> previousErrors;//!!!! size
		vector<Layer*> layers = netToTrain.get_layers();
		vector<double> input(28 * 28);
		unsigned char* copy_data_addr = data_addr;
		unsigned char* copy_activations_addr = activations_addr;

		for (size_t i = 0; i < epochs; i++)
		{
			data_addr = copy_data_addr;
			activations_addr = copy_activations_addr;
			std::cout << "current epoch: " << i << std::endl;
			for (size_t j = 0; j < 10000; j++)
			{
				for (size_t k = 0; k < 28 * 28; k++)
				{
					input[k] = shrinky(*data_addr);
					data_addr++;
				}
				netToTrain.feedForward(input);
				vector<double> correctActivations(layers.back()->getSize(), 0);
				
				correctActivations[*activations_addr] = 1;
				layers[layers.size() - 1]->lastLayerDelta(correctActivations);
				
				for (size_t k = layers.size() - 1; k > 0; k--) {
					previousErrors = layers[k]->layerDelta(layers[k - 1]);// ?
					layers[k - 1]->set_errors(previousErrors);
				}
				activations_addr++;
				//std::cout << j <<  " inner cost: " << this->costFunction() << std::endl;
				//std::cout << j << endl;
			}
			std::cout << "cost: " << this->costFunction() << std::endl;
		}
}

double Trainer::costFunction()
{
	vector<double> errors = netToTrain.get_layers().back()->get_errors();
	for_each(errors.begin(), errors.end(), [](double &elem) {
		elem *= elem;
	});
	
	return accumulate(errors.begin(), errors.end(), 0.0);
}

void Trainer::loadTrainingData(const char * filename)
{
	file_mapping m_data(filename, read_only);//access rights
	region = new mapped_region (m_data, read_only);
	
	data_addr = static_cast<unsigned char*>(region->get_address());
	data_addr += 16;
	data_size = region->get_size();
}

void Trainer::loadCorrectActivations(const char * filename)
{
	file_mapping m_activations(filename, read_only);//access rights
	mapped_region* region2 = new mapped_region(m_activations, read_only);

	activations_addr = static_cast<unsigned char*>(region2->get_address());
	activations_addr += 8;
	activations_size = region2->get_size();
}

unsigned char * Trainer::get_data_addr()
{
	return data_addr;
}

unsigned char * Trainer::get_activations_addr()
{
	return activations_addr;
}

void Trainer::set_data_addr(size_t  offset)
{
	data_addr += offset;
}
