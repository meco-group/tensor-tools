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


  const T& data() const { return data_; }
  int numel() const { return data_.numel(); }

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

  static T get(const T& data, const std::vector<int> dims, const std::vector<int>& ind) {
    return data.nz(ind2sub(dims, ind));
  }

  static void set(T& data, const std::vector<int> dims, const std::vector<int>& ind,
                  const T& rhs) {
    data.nz(ind2sub(dims, ind)) = rhs;
  }

  int n_dims() const {return dims_.size(); }
  const std::vector<int>& dims() const { return dims_; }
  const int dims(int i) const { return dims_[i]; }

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
    // Check that input is a permutaion of range(n_dims())
    assert(order.size()==n_dims());

    std::vector<bool> occured(n_dims(), false);

    for (int i : order) {
      assert(i>=0);
      assert(i<n_dims());
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

  /** \brief Perform a matrix product on the first two indices */
  Tensor partial_product(const Tensor & b) {
    const Tensor& a = *this;

    assert(b.n_dims()>=2);
    assert(a.n_dims()>=2);

    bool fixed = (a.n_dims()==2);

    if (!fixed) {
      for (int i=2;i<a.n_dims();i++) assert(a.dims(i)==b.dims(i));
    }

    assert(b.dims(1)==a.dims(0));

    std::vector<int> new_dims = b.dims();
    new_dims[0] = a.dims(0);

    int trailing_size = 1;
    std::vector<int> trailing_dim(b.dims().begin()+2, b.dims().end());
    for (int i=2;i<b.n_dims();i++) trailing_size*= b.dims(i);

    std::cout << trailing_size << std::endl;

    T data = T::zeros(normalize_dim(new_dims));

    for (int z=0;z<trailing_size;++z) {
      std::vector<int> trailing_ind = sub2ind(trailing_dim, z);
      for (int i=0;i<a.dims(0);++i) {
        for (int j=0;j<a.dims(1);++j) {
          for (int k=0;k<b.dims(2);++k) {
            std::vector<int> a_ind = {i, j};
            if (!fixed) a_ind.insert(a_ind.end(), trailing_ind.begin(), trailing_ind.end());
            std::vector<int> b_ind = {j, k};
            b_ind.insert(b_ind.end(), trailing_ind.begin(), trailing_ind.end());
            std::vector<int> c_ind = {i, k};
            c_ind.insert(c_ind.end(), trailing_ind.begin(), trailing_ind.end());
            T temp = get(data, new_dims, c_ind) + get(data_, dims(), a_ind)*get(b.data_, b.dims(), b_ind);
            set(data, new_dims, c_ind, temp);
          }
        }
      }
    }

    return Tensor(data, new_dims);
  }

  private:
    const T data_;
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
