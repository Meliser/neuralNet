#pragma once
#include<iostream>
#include<boost/archive/text_oarchive.hpp>
#include<boost/archive/text_iarchive.hpp>
using namespace std;
class inputNeuron {
public:
	double get_activation();
	void set_activation(double act);
	virtual ~inputNeuron();
protected:
	double activation;
private:
	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive & ar, const unsigned int version)
	{
		ar & activation;
	}
};