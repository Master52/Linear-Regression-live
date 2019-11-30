#include<iostream>
#include <fstream>
#include <random>
#include <unistd.h>
#include "ML.hpp"

typedef std::pair<Input*,Label*> RegressionData;

static void write(Input* x,Label* y,const char* filename)
{
    std::ofstream fp(filename);
    if(!fp.is_open())
        std::runtime_error("Error: Cannot  write to data\n");

    Input::iterator x_it = x->begin();
    Input::iterator y_it = y->begin();

    for (; x_it != x->end();++x_it,++y_it)
    {
        fp << *x_it;
        fp << ",";
        fp << *y_it;
        fp << "\n";
    }
}

void free_mem(RegressionData* mem)
{
    delete mem->first;
    delete mem->second;
    delete mem;

}


int main(int argc,char* argv[])
{
    RegressionData *data = new RegressionData();
    data->first = new Input();
    data->second = new Label();
    LinearRegression regression;
    int x, y;

    std::random_device rd;
    std::cout << "Enter values of X and Y\n";


    system("gnuplot ./plot.p  > /dev/null 2>&1 & ");
    for (;;)
    {
    	    std::cin >> x >> y;
            data->first->emplace_back(x);
            data->second->emplace_back(y);
            regression.fit(data->first, data->second);
            Output *pred = regression.pridict(data->first);

            write(data->first, data->second, argv[1]);
            write(data->first, pred, argv[2]);
        }

    free_mem(data);
}
