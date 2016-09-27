#include "any_tensor.hpp"



bool AnyScalar::isDouble() const {
  return t == TENSOR_DOUBLE;
}
bool AnyScalar::isSX() const {
  return t == TENSOR_SX;
}
bool AnyScalar::isMX() const {
  return t == TENSOR_MX;
}

bool AnyTensor::isDouble() const {
  return t == TENSOR_DOUBLE;
}
bool AnyTensor::isSX() const {
  return t == TENSOR_SX;
}
bool AnyTensor::isMX() const {
  return t == TENSOR_MX;
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
  return DT(v, {v.size()});
}

/**
AnyTensor::AnyTensor(const std::vector<AnyScalar>&v, const std::vector<int>& dim) {
  std::vector<>
}
*/


AnyTensor vertcat(const std::vector<AnyScalar> & v) {
  return AnyTensor::vertcat(v);
}
