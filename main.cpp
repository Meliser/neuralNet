
#include <iostream>
#include<vector>
#include<initializer_list>
#include"net.h"
#include "Timer.h"
using namespace std;
#define TRAN
int main()
{
	setlocale(0, "rus");
	{
		Timer timer(__FUNCTION__);

		net n = { 2,3,1 };
#ifdef TRAIN
		n.firstInit();
		for (size_t i = 0; i < 500; i++)
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
		saveInFile(n, "data.txt");

#else
		loadFromFile(n, "data.txt");
		vector<double> input3 = { 1,0 };
		n.feedForward(input3);
		const vector<double> res = n.getResult();// not efficient
		for (auto i : res) {
			cout << i << " ";
		}
		cout << endl;
#endif
	}

	

	cout << endl;
	system("pause");
	return 0;
}

