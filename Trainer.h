#pragma once
#include "net.h"
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>
using namespace boost::interprocess;
double shrinky(double x, double in_min, double in_max, double out_min, double out_max);
class Trainer
{
public:
	Trainer(net &obj, size_t _epochs);
	~Trainer();
	void backPropogation();
	double costFunction();
	void loadTrainingData(const char* filename);
	void loadCorrectActivations(const char* filename);
	unsigned char* get_data_addr();
	unsigned char* get_activations_addr();
	void set_data_addr(size_t offset);
private:
	net &netToTrain;
	size_t epochs;
	unsigned char* data_addr;
	unsigned char* activations_addr;
	size_t images;
	size_t data_size;
	size_t activations_size;
	mapped_region* region;
	mapped_region* region2;
};

