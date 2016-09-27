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
  return data_double->outer_product(*b.data_double);
}

std::vector<int> AnyTensor::dims() const {
  switch(t) {
    case TENSOR_DOUBLE:
      return data_double->dims();
      break;
    case TENSOR_SX:
      return data_sx->dims();
      break;
    case TENSOR_MX:
      return data_mx->dims();
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

bool is_double(const std::vector<AnyScalar>& v) {
  bool ret = true;
  for (auto &i : v) {
    ret &= i.is_double();
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

AnyTensor& AnyTensor::operator=(const AnyTensor& s) {
  t = s.t;
  data_double = s.data_double;
  data_sx = s.data_sx;
  data_mx = s.data_mx;
  return *this;
}

AnyTensor::AnyTensor(const AnyTensor& s) {
  t = s.t;
  data_double = s.data_double;
  data_sx = s.data_sx;
  data_mx = s.data_mx;
}

AnyTensor::AnyTensor(const DT & s) {
  t = TENSOR_DOUBLE;
  data_double = new DT(s);
  data_sx = 0;
  data_mx = 0;
}

//AnyTensor::AnyTensor(const AnyScalar& s) {
//  if (s.t == TENSOR_DOUBLE) {
//
//  }
//}

AnyTensor::AnyTensor() {
  t = TENSOR_NULL;
  data_double = 0;
  data_sx = 0;
  data_mx = 0;
}

AnyTensor::operator DT() const {
  assert(TENSOR_DOUBLE);
  return DT(*data_double);
}

AnyTensor::~AnyTensor() {
  if (data_double) delete data_double;
  if (data_sx) delete data_sx;
  if (data_mx) delete data_mx;
}


AnyTensor AnyTensor::vertcat(const std::vector<AnyScalar>& v) {
  std::vector<double> ret;
  ret.reserve(v.size());
  for (auto & i : v) {
    ret.push_back(i);
  }
  return DT(DM(ret), {v.size()});
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
