#include "any_tensor.hpp"

int product(const std::vector<int>& a) {
  int r = 1;
  for (int i=0;i<a.size();++i) r*=a[i];
  return r;
}

bool AnyScalar::is_double() const {
  return t == TENSOR_DOUBLE;
}
bool AnyScalar::is_SX() const {
  return t == TENSOR_SX;
}
bool AnyScalar::is_MX() const {
  return t == TENSOR_MX;
}

bool AnyTensor::is_double() const {
  return t == TENSOR_DOUBLE;
}
bool AnyTensor::is_SX() const {
  return t == TENSOR_SX;
}
bool AnyTensor::is_MX() const {
  return t == TENSOR_MX;
}

AnyTensor AnyTensor::unity() {
  return DT(1, {});
}


AnyTensor AnyTensor::reorder_dims(const std::vector<int>& order) const {
  if (is_double()) return as_DT().reorder_dims(order);
  if (is_SX()) return as_ST().reorder_dims(order);
  if (is_MX()) return as_MT().reorder_dims(order);
  tensor_assert(false);
  return DT();
}
AnyTensor AnyTensor::shape(const std::vector<int>& dims) const {
  if (is_double()) return as_DT().shape(dims);
  if (is_SX()) return as_ST().shape(dims);
  if (is_MX()) return as_MT().shape(dims);
  tensor_assert(false);
  return DT();
}
    
    
TensorType AnyScalar::merge(TensorType a, TensorType b) {

  if (a == TENSOR_SX && b == TENSOR_MX) tensor_assert(0);
  if (a == TENSOR_MX && b == TENSOR_SX) tensor_assert(0);
  if (a == TENSOR_SX || b == TENSOR_SX) return TENSOR_SX;
  if (a == TENSOR_MX || b == TENSOR_MX) return TENSOR_MX;

  return TENSOR_DOUBLE;
}


AnyScalar& AnyScalar::operator+=(const AnyScalar& rhs) {
  AnyScalar ret;
  switch (AnyScalar::merge(t, rhs.t)) {
    case TENSOR_DOUBLE:
      ret = as_double()+rhs.as_double();
      break;
    case TENSOR_SX:
      ret = as_SX()+rhs.as_SX();
      break;
    case TENSOR_MX:
      ret = as_MX()+rhs.as_MX();
      break;
    default: tensor_assert(false);
  }

  return this->operator=(ret);
}

AnyTensor AnyTensor::solve(const AnyTensor& A, const AnyTensor& B) {
  AnyTensor ret;
  switch (AnyScalar::merge(A.t, B.t)) {
    case TENSOR_DOUBLE:
      ret = solve(A.as_DT(),B.as_DT());
      break;
    case TENSOR_SX:
      ret = solve(A.as_ST(),B.as_ST());
      break;
    case TENSOR_MX:
      ret = solve(A.as_MT(),B.as_MT());
      break;
    default: tensor_assert(false);
  }

  return ret;
}


AnyTensor AnyTensor::outer_product(const AnyTensor &b) {
  switch (AnyScalar::merge(t, b.t)) {
    case TENSOR_DOUBLE:
      return data_double.outer_product(b.data_double);
      break;
    case TENSOR_SX:
      return as_ST().outer_product(b.as_ST());
      break;
    case TENSOR_MX:
      return as_MT().outer_product(b.as_MT());
      break;
    default: tensor_assert(false);
  }
  return DT();
}

AnyTensor AnyTensor::inner(const AnyTensor &b) {
  switch (AnyScalar::merge(t, b.t)) {
    case TENSOR_DOUBLE:
      return data_double.inner(b.data_double);
      break;
    case TENSOR_SX:
      return as_ST().inner(b.as_ST());
      break;
    case TENSOR_MX:
      return as_MT().inner(b.as_MT());
      break;
    default: tensor_assert(false);
  }
  return DT();
}

std::vector<int> AnyTensor::dims() const {
  switch (t) {
    case TENSOR_DOUBLE:
      return data_double.dims();
      break;
    case TENSOR_SX:
      return data_sx.dims();
      break;
    case TENSOR_MX:
      return data_mx.dims();
      break;
    default: tensor_assert(false);
  }
  return {};
}

/**bool AnyTensor::equals(const AnyTensor& rhs) const {
  if (dims()==rhs.dims()) {
    if (TENSOR_DOUBLE) {
      return static_cast<double>(norm_inf((*this)-rhs))==0;
    }
  }
  return false;
}*/

std::vector<double> AnyScalar::vector_double(const std::vector<AnyScalar>& v) {
  std::vector<double> ret;
  ret.reserve(v.size());
  for (auto &i : v) {
    ret.push_back(i.as_double());
  }
  return ret;
}

AnyScalar pow(const AnyScalar&x, int i) {
  if (x.is_double()) return pow(x.as_double(), i);
  if (x.is_SX()) return pow(x.as_SX(), SX(i));
  if (x.is_MX()) return pow(x.as_MX(), MX(i));
  tensor_assert(false);
  return 0;
}

bool AnyScalar::is_double(const std::vector<AnyScalar>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_double();
  }
  return ret;
}

bool AnyScalar::is_SX(const std::vector<AnyScalar>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_SX();
  }
  return ret;
}

bool AnyScalar::is_MX(const std::vector<AnyScalar>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_MX();
  }
  return ret;
}

bool AnyTensor::is_DT(const std::vector<AnyTensor>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_double();
  }
  return ret;
}

bool AnyTensor::is_ST(const std::vector<AnyTensor>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_SX();
  }
  return ret;
}

bool AnyTensor::is_MT(const std::vector<AnyTensor>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_MX();
  }
  return ret;
}

AnyScalar::AnyScalar(const AnyScalar& s) {
  t = s.t;
  data_double = s.data_double;
  data_sx = s.data_sx;
  data_mx = s.data_mx;
}

AnyScalar& AnyScalar::operator=(const AnyScalar& s) {
  t = s.t;
  data_double = s.data_double;
  data_sx = s.data_sx;
  data_mx = s.data_mx;
  return *this;
}

AnyScalar::AnyScalar(double s) {
  t = TENSOR_DOUBLE;
  data_double = s;
}

AnyScalar::AnyScalar(const SX& s) {
  t = TENSOR_SX;
  data_sx = s;
}

AnyScalar::AnyScalar(const MX& s) {
  t = TENSOR_MX;
  data_mx = s;
}

AnyScalar::AnyScalar() {
  t = TENSOR_NULL;
  data_double = 0;
  data_sx = 0;
  data_mx = 0;
}

AnyScalar::operator double() const {
  tensor_assert(t==TENSOR_DOUBLE);
  return data_double;
}

AnyScalar::operator SX() const {
  if (t==TENSOR_DOUBLE) return SX(data_double);
  tensor_assert(t==TENSOR_SX);
  return data_sx;
}

AnyScalar::operator MX() const {
  if (t==TENSOR_DOUBLE) return MX(data_double);
  tensor_assert(t==TENSOR_MX);
  return data_mx;
}

AnyTensor& AnyTensor::operator=(const AnyTensor& s) {
  t = s.t;
  data_double = s.data_double;
  data_sx = s.data_sx;
  data_mx = s.data_mx;
  return *this;
}

AnyTensor::AnyTensor(const AnyTensor& s) : data_double(0), data_sx(0), data_mx(0) {
  t = s.t;
  data_double = s.data_double;
  data_sx = s.data_sx;
  data_mx = s.data_mx;
}

AnyTensor::AnyTensor(const DT & s) : data_double(s), data_sx(0), data_mx(0) {
  t = TENSOR_DOUBLE;
}

AnyTensor::AnyTensor(const ST & s) : data_double(0), data_sx(s), data_mx(0) {
  t = TENSOR_SX;
}

AnyTensor::AnyTensor(const MT & s) : data_double(0), data_sx(0), data_mx(s) {
  t = TENSOR_MX;
}

//AnyTensor::AnyTensor(const AnyScalar& s) {
//  if (s.t == TENSOR_DOUBLE) {
//
//  }
//}

AnyTensor::AnyTensor() : data_double(0), data_sx(0), data_mx(0) {
  t = TENSOR_NULL;
}

AnyTensor::operator DT() const {
  tensor_assert(t==TENSOR_DOUBLE);
  return data_double;
}

AnyTensor::operator ST() const {
  if (t==TENSOR_DOUBLE) return ST(data_double);
  tensor_assert(t==TENSOR_SX);
  return data_sx;
}

AnyTensor::operator MT() const {
  if (t==TENSOR_DOUBLE) return MT(data_double);
  tensor_assert(t==TENSOR_MX);
  return data_mx;
}


AnyTensor AnyTensor::concat(const std::vector<AnyTensor>& v, int axis) {
  tensor_assert_message(false, "Not implemented");
  return DT();
}
    
AnyTensor AnyTensor::pack(const std::vector<AnyTensor>& v, int axis) {
  if (AnyTensor::is_DT(v)) {
    std::vector<DT> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i.as_DT());
    }
    return DT::pack(ret, axis);
  }
  if (AnyTensor::is_ST(v)) {
    std::vector<ST> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i.as_ST());
    }
    return ST::pack(ret, axis);
  }
  if (AnyTensor::is_MT(v)) {
    std::vector<MT> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i.as_MT());
    }
    return MT::pack(ret, axis);
  }
  tensor_assert(false);
  return DT();
}

std::vector<AnyTensor> unpack(const AnyTensor& v, int axis) {
  tensor_assert_message(false, "Not implemented yet");
  return {DT()};
}
    
    
AnyTensor AnyTensor::vertcat(const std::vector<AnyScalar>& v) {
  if (AnyScalar::is_double(v)) {
    std::vector<double> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i.as_double());
    }
    return DT(DM(ret), {static_cast<int>(v.size())});
  }
  if (AnyScalar::is_SX(v)) {
    std::vector<SX> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i.as_SX());
    }
    return ST(SX::vertcat(ret), {static_cast<int>(v.size())});
  }
  if (AnyScalar::is_MX(v)) {
    std::vector<MX> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i.as_MX());
    }
    return MT(MX::vertcat(ret), {static_cast<int>(v.size())});
  }
  tensor_assert(false);
  return DT();
}




/**
AnyTensor::AnyTensor(const std::vector<AnyScalar>&v, const std::vector<int>& dim) {
  std::vector<>
}
*/


AnyTensor vertcat(const std::vector<AnyScalar> & v) {
  return AnyTensor::vertcat(v);
}

AnyTensor concat(const std::vector<AnyTensor> & v, int axis) {
  return AnyTensor::concat(v, axis);
}


AnyTensor vertcat(const std::vector<double>& v) {
  return DT(DM(v), {static_cast<int>(v.size())});
}
