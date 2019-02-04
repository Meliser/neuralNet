#pragma once
#include<vector>
//#include"inputNeuron.h"
//#include"neuron.h"
using namespace std;
template<typename T>
class Layer
{
public:
	Layer() {};
	~Layer() {};

private:
	std::vector<T> neurons;
};

//typename Layer<int>