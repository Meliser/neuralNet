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
	try {
		net n = { 784,16,16,10 };
#ifdef INIT
		n.firstInit();
		saveInFile(n, "data.txt");
#endif
#ifdef TRAIN
		Trainer trainer(n, 3);

		loadFromFile(n, "data.txt");

		trainer.loadTrainingData("t10k-images.idx3-ubyte");// t10k-images.idx3-ubyte images60000.idx3-ubyte
		trainer.loadCorrectActivations("t10k-labels.idx1-ubyte");// t10k-labels.idx1-ubyte labels60000.idx1-ubyte

		trainer.backPropogation();

		saveInFile(n, "data.txt");
#endif
#ifdef LOAD
		loadFromFile(n, "data.txt");
		Trainer t(n, 1);
		t.loadTrainingData("t10k-images.idx3-ubyte");
		cout << "Выберите одну из 10000 цифр для проверки нейросети(индексация с 0): ";
		int digit;
		cin >> digit;
		t.set_data_addr(784 * digit);
		vector<double> input3(784);
		for (size_t i = 0; i < 28; i++)
		{
			for (size_t j = 0; j < 28; j++)
			{
				input3[28 * i + j] = shrink(*t.get_data_addr(), 0, 255, 0, 1);
				if (input3[28 * i + j] != 0) {
					cout << "* ";
				}
				else {
					cout << 0 << " ";
				}
				t.set_data_addr(1);
			}
			cout << endl;
		}

		n.feedForward(input3);
		const vector<double> &res = n.getResult();// not efficient
		size_t num = 0;
		cout << endl << "Предположения нейросети" << endl;
		for (auto i : res) {
			cout << num << ": " << i << endl;
			num++;
		}
		cout << endl << endl;
#endif

	}
	catch (std::exception &ex) {
		cout << ex.what();
			cout << endl;
		system("pause");
	}
	cout << endl;
	system("pause");
	return 0;
}
// Добавить многопоточность

