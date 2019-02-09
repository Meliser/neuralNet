#pragma once
#include<iostream>
#include<vector>
#include<initializer_list>
#include"Layer.h"

class net {
public:
	net(initializer_list<int> &&ls);
	~net();
	void feedForward(vector<double> &inputActivations);
	void backPropogation(vector<double> &correctActivations);
private:
	static double learningRate;
	vector<Layer*> layers;
};