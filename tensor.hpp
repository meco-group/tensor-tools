#include <iostream>
#include <fstream>
#include <ctime>
#include <assert.h>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace std;

template <class T>
class Tensor {
  public:

  Tensor(const T& data, const std::vector<int>& dims) : data_(data), dims_(dims) {

  }

  static std::pair<int,int> normalize_dim(const std::vector<int> & dims);
  static void assert_match_dim(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a==b);

  }

  static Tensor sym(const std::string& name, const std::vector<int> & dims) {
    T v = T::sym(name, normalize_dim(dims));
    return Tensor<T>(v, dims);
  }

  Tensor operator+(const Tensor& rhs) const {
    assert_match_dim(dims_, rhs.dims_);
    return Tensor(data_+rhs.data_, dims_);
  }

  Tensor operator*(const Tensor& rhs) const {
    assert_match_dim(dims_, rhs.dims_);
    return Tensor(data_*rhs.data_, dims_);
  }

  T data_;

  private:
    std::vector<int> dims_;
};

template <class T>
std::pair<int,int> Tensor<T>::normalize_dim(const std::vector<int> & dims) {
    if (dims.size()==2) {
      return std::pair<int,int>({dims[0],dims[1]});
    } else if (dims.size()==1) {
      return std::pair<int,int>({dims[0],1});
    } else if (dims.size()>2) {
      int prod = 1;
      for (int i=1;i<dims.size();i++) {
        prod*= dims[i];
      }
      return std::pair<int,int>({dims[0],prod});
    }
}

typedef Tensor<SX> ST;
typedef Tensor<DM> DT;
