
#include <iostream>
#include<vector>
#include<initializer_list>
#include"net.h"
#include "Trainer.h"
#include "Timer.h"
using namespace std;


//#define TRAIN
//#define INIT
#define LOAD
int main()
{
	setlocale(0, "rus");
	{
		Timer timer(__FUNCTION__);

		net n = { 784,16,16,10 };
#ifdef INIT
		n.firstInit();
		saveInFile(n, "data.txt");
#endif
#ifdef TRAIN
		Trainer trainer(n,2);
		
		loadFromFile(n, "data.txt");
		//cout << trainer.costFunction() << endl;
		trainer.loadTrainingData("t10k-images.idx3-ubyte");// t10k-images.idx3-ubyte images60000.idx3-ubyte
		trainer.loadCorrectActivations("t10k-labels.idx1-ubyte");// t10k-labels.idx1-ubyte labels60000.idx1-ubyte
		//cout << trainer.costFunction() << endl;
		trainer.backPropogation();
		/*for (size_t i = 0; i < 500; i++)
		{

			vector<double> input = { 0.99,0 };
			n.feedForward(input);
			vector<double> trainer0 = { 0,1 };
			trainer.backPropogation(trainer0);

			vector<double> input1 = { 0.99,0.99 };
			n.feedForward(input1);
			vector<double> trainer1 = { 0,0 };
			trainer.backPropogation(trainer1);

			vector<double> input2 = { 0,0 };
			n.feedForward(input2);
			vector<double> trainer2 = { 1,1 };
			trainer.backPropogation(trainer2);
		}*/
		saveInFile(n, "data.txt");
#endif
#ifdef LOAD
		loadFromFile(n, "data.txt");
		Trainer t(n, 1);
		t.loadTrainingData("t10k-images.idx3-ubyte");
		t.set_data_addr(0);
		vector<double> input3(784);
		for (size_t i = 0; i < 28; i++)
		{
			for (size_t j = 0; j < 28; j++)
			{
				input3[28*i+j] = shrinky(*t.get_data_addr(), 0, 255, 0, 1);
				if (input3[28 * i + j] != 0) {
					cout << "* ";
				}
				else {
					cout << 0<<" ";
				}
				t.set_data_addr(1);
			}
			cout << endl;
		}
		
		n.feedForward(input3);
		const vector<double> &res = n.getResult();// not efficient
		size_t num=0;
		for (auto i : res) {
			cout <<num<<": "<<i << endl;
			num++;
		}
		cout << endl<<endl;
#endif

	}
	cout << endl;
	system("pause");
	return 0;
}
// Добавить многопоточность

