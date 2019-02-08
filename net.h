#pragma once
#include<iostream>
#include<vector>
#include<initializer_list>
#include"neuron.h"
#include"Layer.h"

class net {
public:
	net(initializer_list<int> &&ls);
	~net();
	void feedForward(vector<double> &inputActivations);
	void backPropogation(vector<double> &correctActivations);
private:
	//vector<vector<inputNeuron*>> layers;
	static double learningRate;
	//inputLayer *input_layer;
	vector<Layer*> layers;
};