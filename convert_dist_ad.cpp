// Convert return time probabilities, X[t], into the first return time
// distribution Y[t]. 
// -GTC 2/5/24
#include </opt/homebrew/Cellar/boost/1.87.0/include/boost/math/differentiation/autodiff.hpp>

extern "C" {
  void convert_dist(double *X, double *Y) {
    auto z = boost::math::differentiation::make_fvar<double, ORD>(0);
    auto zk = z;
    auto R = 1 + (0*zk);
    for (int k=2; k<ORD; k++) R += X[k] * (zk*=z);
    R = 1 - (1/R);
    for (int k=0; k<ORD; k++) Y[k] = R[k];
  }
}
