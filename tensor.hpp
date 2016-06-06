#include <iostream>
#include <fstream>
#include <ctime>
#include <assert.h>
#include <casadi/casadi.hpp>

using namespace casadi;
using namespace std;

template <class T>
std::vector<T> reorder(const std::vector<T>& data, const std::vector<int>& order) {
  std::vector<T> ret(data.size());
  for (int k=0;k<order.size();k++) {
    ret[k] = data[order[k]];
  }
  return ret;
}

template <class T>
class Tensor {
  public:

  Tensor(const T& data, const std::vector<int>& dims) : data_(data), dims_(dims) {
    assert(data.is_dense());
  }

  static std::pair<int, int> normalize_dim(const std::vector<int> & dims);

  static void assert_match_dim(const std::vector<int>& a, const std::vector<int>& b) {
    assert(a==b);

  }

  static std::vector<int> sub2ind(const std::vector<int>& dims, int sub) {
    std::vector<int> ret(dims.size());
    int cumprod = 1;
    for (int i=0;i<dims.size();i++) {
      ret[i] = sub % dims[i];
      sub/= dims[i];
    }
    return ret;
  }
  static int ind2sub(const std::vector<int>& dims, const std::vector<int>& ind) {
    assert(dims.size()==ind.size());
    int ret=0;
    int cumprod = 1;
    for (int i=0;i<dims.size();i++) {
      ret+= ind[i]*cumprod;
      cumprod*= dims[i];
    }
    return ret;
  }

  int dims() const {return dims_.size(); }

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

  /** \brief Generalization of transpose
  */
  Tensor reorder_dims(const std::vector<int>& order) const {
    // Check that input is a permutaion of range(dims())
    assert(order.size()==dims());

    std::vector<bool> occured(dims(), false);

    for (int i : order) {
      assert(i>=0);
      assert(i<dims());
      occured[i] = true;
    }

    for (bool occ : occured) {
      assert(occ);
    }

    T new_data = data_;

    std::vector<int> new_dims = reorder(dims_, order);

    for (int i=0;i<data_.numel();i++) {
      std::vector<int> orig_indices = sub2ind(dims_, i);
      std::vector<int> new_indices = reorder(orig_indices, order);
      int j = ind2sub(new_dims, new_indices);
      new_data.nz(j) = data_.nz(i);
    }

    return Tensor(new_data, new_dims);
  }

  const T data_;

  private:
    const std::vector<int> dims_;
};

template <class T>
std::pair<int, int> Tensor<T>::normalize_dim(const std::vector<int> & dims) {
    if (dims.size()==2) {
      return std::pair<int, int>({dims[0], dims[1]});
    } else if (dims.size()==1) {
      return std::pair<int, int>({dims[0], 1});
    } else if (dims.size()>2) {
      int prod = 1;
      for (int i=1;i<dims.size();i++) {
        prod*= dims[i];
      }
      return std::pair<int, int>({dims[0], prod});
    }
}

typedef Tensor<SX> ST;
typedef Tensor<DM> DT;
