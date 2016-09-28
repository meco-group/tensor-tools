#include "any_tensor.hpp"



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

AnyTensor AnyTensor::outer_product(const AnyTensor &b) {
  switch (t) {
    case TENSOR_DOUBLE:
      return data_double.outer_product(b.data_double);
      break;
    case TENSOR_SX:
      //return data_sx->outer_product(*b.data_sx);
      break;
    case TENSOR_MX:
      //return data_mx->outer_product(*b.mx);
      break;
  }
}

AnyTensor AnyTensor::inner(const AnyTensor &b) {
  switch (t) {
    case TENSOR_DOUBLE:
      return data_double.inner(b.data_double);
      break;
    case TENSOR_SX:
      //return data_sx->inner(*b.data_sx);
      break;
    case TENSOR_MX:
      //return data_mx->inner(*b.mx);
      break;
  }
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
  }
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
    ret.push_back(i);
  }
  return ret;
}

AnyScalar pow(const AnyScalar&x, int i) {
  if (x.is_double()) return pow(x.as_double(), i);
  if (x.is_SX()) return pow(x.as_SX(), SX(i));
  if (x.is_MX()) return pow(x.as_MX(), MX(i));
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
  assert(TENSOR_DOUBLE);
  return data_double;
}

AnyScalar::operator SX() const {
  assert(TENSOR_DOUBLE);
  return data_sx;
}

AnyScalar::operator MX() const {
  assert(TENSOR_SX);
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
  assert(TENSOR_DOUBLE);
  return data_double;
}

AnyTensor::operator ST() const {
  assert(TENSOR_SX);
  return data_sx;
}

AnyTensor::operator MT() const {
  assert(TENSOR_MX);
  return data_mx;
}


AnyTensor AnyTensor::vertcat(const std::vector<AnyScalar>& v) {
  if (AnyScalar::is_double(v)) {
    std::vector<double> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i);
    }
    return DT(DM(ret), {v.size()});
  }
  if (AnyScalar::is_SX(v)) {
    std::vector<SX> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i);
    }
    return ST(SX::vertcat(ret), {v.size()});
  }
  if (AnyScalar::is_MX(v)) {
    std::vector<MX> ret;
    ret.reserve(v.size());
    for (auto & i : v) {
      ret.push_back(i);
    }
    return MT(MX::vertcat(ret), {v.size()});
  }
}



/**
AnyTensor::AnyTensor(const std::vector<AnyScalar>&v, const std::vector<int>& dim) {
  std::vector<>
}
*/


AnyTensor vertcat(const std::vector<AnyScalar> & v) {
  return AnyTensor::vertcat(v);
}


AnyTensor vertcat(const std::vector<double>& v) {
  return DT(DM(v), {v.size()});
}
