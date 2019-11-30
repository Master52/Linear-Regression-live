#include "ML.hpp"


LinearRegression::LinearRegression()
{
    pred = nullptr;
}

double LinearRegression::mean(std::vector<double>* arr)
{
    long long sum = 0;
    for(auto& val : *arr)
        sum += val;

    return sum / arr->size();
}
void LinearRegression::fit(Input* X,Label* Y)
{
    if( X == nullptr || Y == nullptr )
        std::runtime_error("Inpout or Output are not null");

    x_mean = mean(X);
    y_mean = mean(Y);

    /*numerator : (xi - mean(x))  (yi - mean(y)) */
    /*denomentor : (xi - mean(x))^2  */

    double numerator = 0;
    double denomenator = 0;

    Input::iterator xi = X->begin();
    Label::iterator yi = Y->begin();
    for (; xi != X->end(); ++xi,++yi) {
            numerator += (*xi - x_mean) * (*yi - y_mean);
            denomenator += std::pow((*xi - x_mean), 2);
    }

    m = numerator / denomenator;
    b = y_mean - (m  *x_mean);

#if DEBUG 
    std::cerr << "bias(intercept): " << m << "\n"
              << "weight(slope):  " << b << "\n";
#endif
}


Output* LinearRegression::pridict(Label* test)
{
    pred = new Output();
    if(pred == nullptr)
        std::__throw_bad_alloc();

    for (auto &val : *test)
        pred->emplace_back(((m * val) + b));

    return pred;
}

LinearRegression::~LinearRegression()
{
    delete pred;
}


/*************************************************/

LinearGredientDecent::LinearGredientDecent() {}


void LinearGredientDecent::fit(Input *X, Label *Y,size_t& iterations,double& learner)
{
    /*
        W0 = partial derivative of weight(m or intercept)
        W1 = partial derivative of bias(b or slope)

        W0 = 2/N * sumition[-xi(yi-(weight*xi+bias))]
        W1 = 2/N * sumition[-yi-(weight*xi+bias)]
    */

    size_t N = X->size();

    double m_current = 1;
    double b_current = 0;
    double twoBy_N = (double)2 / N;

    for (size_t i = 0; i < iterations; ++i)
    {
        double m_gradient = 0;
        double b_gradient = 0;
        Input::iterator x_it = X->begin();
        Input::iterator y_it = Y->begin();

        for (; x_it != X->end(); ++x_it, ++y_it)
        {
            double temp = (*y_it) - (m_current * (*x_it) + b_current);
            m_gradient += -(twoBy_N * ((*x_it) * (temp)));
            b_gradient += -(twoBy_N * temp);
            m_current = m_current - (learner * m_gradient);
            b_current = b_current - (learner * b_gradient);

        }

    }
    m = m_current;
    b = b_current;

#ifdef DEBUG
   std::cerr <<"\nm: " << m << "\n"
             << "b: " << b << "\n";
    #endif
}

LinearGredientDecent::~LinearGredientDecent() {}
