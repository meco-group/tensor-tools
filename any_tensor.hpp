#ifndef ANY_TENSOR_HPP_INCLUDE
#define ANY_TENSOR_HPP_INCLUDE

#include "tensor.hpp"

enum TensorType {TENSOR_NULL, TENSOR_DOUBLE, TENSOR_SX, TENSOR_MX};



class AnyScalar {

  public:
    AnyScalar& operator=(const AnyScalar&);
    AnyScalar(const AnyScalar& s);
    AnyScalar(double s);
    AnyScalar();
    operator double() const;
  private:
    TensorType t;
    double data_double;
    SX data_sx;
    MX data_mx;
};

class AnyTensor {
  public:
    AnyTensor& operator=(const AnyTensor&);
    //AnyTensor(const AnyScalar& s);
    AnyTensor(const AnyTensor& s);
    //AnyTensor(const AnyTensor&, const std::vector<int>& dim);
    AnyTensor(const DT & t);
    AnyTensor();
    ~AnyTensor();

    operator DT() const;

    static AnyTensor vertcat(const std::vector<AnyScalar>& v);
  private:
    TensorType t;
    DT* data_double;
    ST* data_sx;
    MT* data_mx;
};


AnyTensor vertcat(const std::vector<AnyScalar> & v);

#endif
