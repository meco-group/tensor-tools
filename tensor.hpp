#ifndef TENSOR_HPP_INCLUDE
#define TENSOR_HPP_INCLUDE

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
std::vector<T> mrange(T stop) {
  return range(-1, -stop-1, -1);
}

template <class T>
std::vector<T> mrange(T start, T stop) {
  return range(-start-1, -stop-1, -1);
}

template <class T>
class Tensor {
  public:

  Tensor(const T& data, const std::vector<int>& dims) : data_(data), dims_(dims) {
    assert(data.is_dense());
  }

  Tensor(const Tensor& t) : data_(t.data()), dims_(t.dims()) {
  }

  ~Tensor() { }

  Tensor& operator=(const Tensor& t) {
    dims_ = t.dims();
    data_ = t.data();
    return *this;
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
    return data[ind2sub(dims, ind)];
  }

  static void set(T& data, const std::vector<int> dims, const std::vector<int>& ind,
                  const T& rhs) {
    data[ind2sub(dims, ind)] = rhs;
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

  /** \brief Make a slice
  *
  *   -1  indicates a slice
  */
  Tensor slice(const std::vector<int>& ind) const {
    // Check that input is a permutation of range(n_dims())
    assert(ind.size()==n_dims());

    std::vector<int> slice_dims;

    std::vector<int> slice_location;

    for (int i=0;i<n_dims();++i) {
      if (ind[i]==-1) {
        slice_dims.push_back(dims(i));
        slice_location.push_back(i);
      } else {
        assert(ind[i]>=0);
        assert(ind[i]<dims(i));
      }
    }

    int c=1;
    std::vector<int> a_e;
    std::vector<int> c_e;
    for (int i=0;i<ind.size();++i) {
      if (ind[i]>=0) {
        a_e.push_back(ind[i]);
      } else {
        a_e.push_back(-c);
        c_e.push_back(-c);
        c+=1;
      }
    }

    return einstein(a_e, c_e);
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

    std::vector<int> ind(order.size());
    for (int i=0;i<ind.size();++i) {
      ind[i] = -order[i]-1;
    }

    return einstein(mrange(n_dims()), ind);
  }

  Tensor einstein(const std::vector<int>& a_e, const std::vector<int>& c_e) const {
    return einstein(Tensor(T::zeros(0, 0), {}), a_e, {}, c_e);
  }

  /** \brief Compute any contraction of two tensors, using index/einstein notation

    A.einstein(B, a, b, c) -> C

    Given two tensors, A and B, computes a third tensor C such that:

    C_c = A_a * B_b

    With a, b, c representing einstein indices.
    Instead of the classical index labels i,j,k,... we employ -1,-2,-3,...

  */
  Tensor einstein(const Tensor &B, const std::vector<int>& a,
      const std::vector<int>& b, const std::vector<int>& c) const {

    const Tensor& A = *this;

    bool has_B = B.n_dims()>0;

    // Dimension check
    assert(A.n_dims()==a.size());
    assert(B.n_dims()==b.size());

    std::map<int, int> dim_map;

    // Check if shared nodes dimensions match up
    for (int i=0;i<a.size();++i) {
      int ai = a[i];
      if (ai>=0) continue;
      auto al = dim_map.find(ai);
      if (al==dim_map.end()) {
        dim_map[ai] = A.dims(i);
      } else {
        assert(al->second==A.dims(i));
      }
    }

    for (int i=0;i<b.size();++i) {
      int bi = b[i];
      if (bi>=0) continue;
      auto bl = dim_map.find(bi);
      if (bl==dim_map.end()) {
        dim_map[bi] = B.dims(i);
      } else {
        assert(bl->second==B.dims(i));
      }
    }

    std::vector<int> new_dims;
    for (int i=0;i<c.size();++i) {
      int ci = c[i];
      assert(ci<0);
      auto cl = dim_map.find(ci);
      assert(cl!=dim_map.end());
      new_dims.push_back(dim_map[ci]);
    }

    T data = T::zeros(normalize_dim(new_dims));

    // Compute the total number of iterations needed
    int n_iter = 1;
    std::vector<int> dim_map_keys;
    std::vector<int> dim_map_values;
    for (const auto& e : dim_map) {
      n_iter*= e.second;
      dim_map_keys.push_back(e.first);
      dim_map_values.push_back(e.second);
    }

    // Main loop
    for (int i=0;i<n_iter;++i) {
      std::vector<int> ind_total = sub2ind(dim_map_values, i);
      std::vector<int> ind_a, ind_b, ind_c;
      int sub_a, sub_b, sub_c;

      for (const auto& ai : a) {
        ind_a.push_back(ai<0 ? ind_total[distance(dim_map.begin(), dim_map.find(ai))] : ai);
      }
      if (has_B) {
        for (const auto& bi : b) {
          ind_b.push_back(bi<0 ? ind_total[distance(dim_map.begin(), dim_map.find(bi))] : bi);
        }
      }
      for (const auto& ci : c) {
        if (ci<0) {
          ind_c.push_back(ind_total[distance(dim_map.begin(), dim_map.find(ci))]);
        }
      }

      sub_a = ind2sub(A.dims(), ind_a);
      if (has_B) sub_b = ind2sub(B.dims(), ind_b);
      sub_c = ind2sub(new_dims, ind_c);
      if (has_B) {
        data[sub_c]+= data_[sub_a]*B.data()[sub_b];
      } else {
        data[sub_c]+= data_[sub_a];
      }
    }

    return Tensor(data, new_dims);
  }

  /**
    c_ijkm = a_ij*b_km
  */
  Tensor outer_product(const Tensor &b) {
    return einstein(b, mrange(n_dims()), mrange(n_dims(), n_dims()+b.n_dims()),
                                         mrange(n_dims()+b.n_dims()));
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

    std::vector<int> a_r;
    if (fixed) {
      a_r = {-1, -b.n_dims()-1};
    } else {
      a_r = mrange(a.n_dims());
      a_r[1] = -b.n_dims()-1;
    }
    std::vector<int> b_r = mrange(b.n_dims());
    b_r[0] = -b.n_dims()-1;
    std::vector<int> c_r = mrange(b.n_dims());

    return einstein(b, a_r, b_r, c_r);
  }

  #ifndef SWIG
  /// Print a representation of the object to a stream (shorthand)
  inline friend
      std::ostream& operator<<(std::ostream &stream, const Tensor& obj) {
          return stream << "Tensor(" << obj.data_.type_name() << ", "
            << obj.dims() << "): " << obj.data();
      }
  #endif // SWIG

  private:
    T data_;
    std::vector<int> dims_;
};

template <class T>
std::pair<int, int> Tensor<T>::normalize_dim(const std::vector<int> & dims) {
    if (dims.size()==0) {
      return {1, 1};
    } else if (dims.size()==2) {
      return {dims[0], dims[1]};
    } else if (dims.size()==1) {
      return {dims[0], 1};
    } else if (dims.size()>2) {
      int prod = 1;
      for (int i=1;i<dims.size();i++) {
        prod*= dims[i];
      }
      return {dims[0], prod};
    }
}

typedef Tensor<SX> ST;
typedef Tensor<DM> DT;
typedef Tensor<MX> MT;

#endif
