#pragma once
#include<iostream>
#include<fstream>
#include<vector>
#include<initializer_list>
#include<boost/archive/text_oarchive.hpp>
#include<boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include"Layer.h"

class net {
public:
	net(initializer_list<int> &&ls);
	~net();
	void feedForward(vector<double> &inputActivations);
	const vector<double> getResult();
	vector<Layer*>& get_layers();
	void firstInit();
private:
	static double learningRate;//add method
	vector<Layer*> layers;
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & layers;
	}
};
void saveInFile(net &obj, const char* filename);// replace in Trainer class
void loadFromFile(net &obj, const char* filename);