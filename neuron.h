#pragma once
#include<iostream>
#include<vector>
#include<math.h>
#include<random>
#include<chrono>
#include<boost/archive/text_oarchive.hpp>
#include<boost/archive/text_iarchive.hpp>
#include"inputNeuron.h"

class neuron :public inputNeuron {
public:
	neuron(size_t weightVectorSize);
	neuron(){}
	virtual ~neuron();
	void activate(vector<inputNeuron*> &previous_neurons);
	double sigmoidDerivativeZ();
	double& getZ();
	vector<double>& getWeights();
	void setWeights(vector<double>& _weights);
	double& getBias();
	void setBias(double _bias);
	void initWeightsRandomly();
private:
	vector<double> weights;
	double bias;
	double z;
	static default_random_engine engine;
	static uniform_real_distribution<> distr;
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & boost::serialization::base_object<inputNeuron>(*this);
		ar & weights & bias & z;// static variables?
	}
};
