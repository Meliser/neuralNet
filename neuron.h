#pragma once
#include<iostream>
#include<vector>
#include<math.h>
#include<random>
#include<chrono>
#include"inputNeuron.h"
//#include"Layer.h"

class neuron :public inputNeuron {
public:
	neuron(size_t weightVectorSize);
	virtual ~neuron();
	void activate(vector<inputNeuron*> &previous_neurons);
	double sigmoidDerivativeZ();
	double& getZ();
	vector<double>& getWeights();
	double& getBias();
private:
	vector<double> weights;
	double bias;
	double z;
	static default_random_engine engine;
	static uniform_real_distribution<> distr;
};
