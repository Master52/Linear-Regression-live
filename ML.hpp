#ifndef __ML__
#define __ML__

#include <vector>
#include <cmath>
#include <iostream>


#define DEBUG 0

typedef std::vector<double> Input;
typedef std::vector<std::vector<double>* > Input2D;
typedef std::vector<double> Label;
typedef std::vector<double> Output;


class LinearRegression
{
    private:
        double mean(std::vector<double> *arr);

    protected:
        Output *pred;

        double x_mean;
        double y_mean;

        double m,b; /* bias: intercept , weight : slop */


    public:
        LinearRegression();
        void fit(Input *X, Label *Y); //Simple linear regression
        Output *pridict(Label *test);
        ~LinearRegression();
};

class LinearGredientDecent:public LinearRegression
{
    private:
    void step(double &twoBy_N, double &learning);

    public:
    LinearGredientDecent();

    void fit(Input *X, Label *Y,size_t& iterattions,double& learner);

    ~LinearGredientDecent();

};

static double SSE(Label* target,Label* pred)
{
    double sum = 0;

    Label::iterator target_it = target->begin();
    Label::iterator pred_it   = pred->begin();

    for (; target_it != target->end(); ++target_it,++pred_it)
        sum += std::pow((*target_it - *pred_it),2);

    return sum;

}

static double RMSE(Label *target, Label *pred)
{

    return (std::sqrt(SSE(target,pred) / target->size()));

}


#endif
