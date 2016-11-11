#include <cmath>

// tanh(x)
double tanH(double x)
{
	return tanh(x);
}
// derivative of tanh(x)
double dtanH(double x)
{
	return 1.0 - tanh(x)*tanh(x);
}

// identity
double identity(double x)
{
	return x;
}
// derivative of identity
double didentity(double x)
{
	return 1.0;
}

// logistic
double logistic(double x)
{
	return 1.0/(1.0+exp(-x));
}
// derivative of logistic
double dlogistic(double x)
{
	return logistic(x)*(1.0-logistic(x));
}
