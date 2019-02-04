#include <iostream>
#include<math.h>
#include<random>
#include<chrono>
#include<vector>
#include<initializer_list>
#include"net.h"
#include"Layer.cpp"
#include"Timer.h"
using namespace std;
// add layer class
// maybe connection class




int main()
{
	setlocale(0, "rus");
	{
		Timer timer;
		//vector<int> topology = { 2,3,1 };
		net n(topology);
		Layer<neuron> a;
		for (size_t i = 0; i < 200; i++)
		{

			vector<double> input = { 0,0 };
			n.feedForward(input);
			vector<double> trainer = { 0 };
			n.backPropogation(trainer);

			vector<double> input1 = { 1,0 };
			n.feedForward(input1);
			vector<double> trainer1 = { 1 };
			n.backPropogation(trainer1);

			vector<double> input2 = { 1,1 };
			n.feedForward(input2);
			vector<double> trainer2 = { 1 };
			n.backPropogation(trainer2);
		}
		
		
		cout << endl;
		vector<double> input3 = { 0,0 };
		n.feedForward(input3);
		//вывод активаций 
	}
	
	cout << endl;
	system("pause");
	return 0;
}
//vector<double> input1 = { 0,1 };
//n.feedForward(input1);
//vector<double> trainer1 = { 1 };
//n.backPropogation(trainer1);
//{
//	vector<double> input = { 0,0 };
//	n.feedForward(input);
//	vector<double> trainer = { 0 };
//	n.backPropogation(trainer);
//}
//vector<double> input3 = { 1,0 };
//n.feedForward(input3);
//vector<double> trainer3 = { 1 };
//n.backPropogation(trainer3);
//{
//	vector<double> input = { 0,0 };
//	n.feedForward(input);
//	vector<double> trainer = { 0 };
//	n.backPropogation(trainer);
//}
//vector<double> input2 = { 1,1 };
//n.feedForward(input2);
//vector<double> trainer2 = { 1 };
//n.backPropogation(trainer2);
//{
//	vector<double> input = { 0,0 };
//	n.feedForward(input);
//	vector<double> trainer = { 0 };
//	n.backPropogation(trainer);
//}
//class neuron {
//public:
//	neuron(double w,double b) :weight(w), bias(b) {}
//	void activate(double previous_act) {
//		z = previous_act * weight + bias;
//		activation = 1 / (1 + exp(-z));// 0..1
//		cout << "activation: " << activation << endl;
//	}
//	double get_activation() {
//		return activation;
//	}
//	void set_activation(double act) {
//		activation = act;
//	}
//	double cost(double trainer) {
//		return (activation - trainer)*(activation - trainer);
//	}
//	double backpropogation(double previous_act, double wanted) {
//		double sigmoid_derivative = 1 / (1 + exp(-z)) * (1 - 1 / (1 + exp(-z)));
//		double weight_delta = 2 * previous_act*sigmoid_derivative*(activation - wanted);
//		double bias_delta = 2 * sigmoid_derivative*(activation - wanted);
//		double previous_activation_delta = 2 * sigmoid_derivative*(activation - wanted)*weight;
//		weight -= weight_delta;
//		bias -= bias_delta;
//		return previous_activation_delta;
//	}
//private:
//	double weight;
//	double bias;
//	double activation;
//	double z;
//};
/*neuron n1(0.6, 2);
	neuron n2(0.5, 1);

	for (size_t i = 0; i < 1000; i++)
	{
		//cout << "n1 ";
		n1.activate(0.3);
		cout << endl;
		cout << "n2 ";
		n2.activate(n1.get_activation());
		cout << endl;
		double act_delta = n2.backpropogation(n1.get_activation(), 0.6);
		double wanted = n1.get_activation() - act_delta;
		n1.backpropogation(0.3, wanted);

		n1.activate(0.7);
		cout << endl;
		cout << "n2 ";
		n2.activate(n1.get_activation());
		cout << endl;
		 act_delta = n2.backpropogation(n1.get_activation(), 0.3);
		 wanted = n1.get_activation() - act_delta;
		n1.backpropogation(0.7, wanted);
	}
	

	n1.activate(0.25);
	cout << "n2 from 0.3 new ";
	n2.activate(n1.get_activation());
	n1.activate(0.);
	cout << "n2 from 0.7 new ";
	n2.activate(n1.get_activation());*/