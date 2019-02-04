#pragma once
#include<iostream>
#include<vector>
#include"neuron.h"

class net {
public:
	net(vector<int> topology);
	~net();
	void feedForward(vector<double> &inputActivations);
	void backPropogation(vector<double> &correctActivations);
private:
	vector<vector<inputNeuron*>> layers;
	static double learningRate;
};
