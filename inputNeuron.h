#pragma once
#include<iostream>
using namespace std;
class inputNeuron {
public:
	double get_activation();
	void set_activation(double act);
	virtual ~inputNeuron();
protected:
	double activation;
};