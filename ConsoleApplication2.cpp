#include <iostream>
#include<vector>
#include<initializer_list>
#include"net.h"
#include "Timer.h"
using namespace std;
int main()
{
	setlocale(0, "rus");
	{
		Timer timer(__FUNCTION__);
		
		net n = { 2,3,1 };
		
		
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
	}
	cout << endl;
	system("pause");
	return 0;
}

