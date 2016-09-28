#ifndef ANY_TENSOR_HPP_INCLUDE
#define ANY_TENSOR_HPP_INCLUDE

#include "tensor.hpp"

enum TensorType {TENSOR_NULL, TENSOR_DOUBLE, TENSOR_SX, TENSOR_MX};



class AnyScalar {

  public:
#ifndef SWIG
    AnyScalar& operator=(const AnyScalar&);
#endif
    AnyScalar(const AnyScalar& s);
    AnyScalar(double s);
    AnyScalar(const SX& s);
    AnyScalar(const MX& s);
    AnyScalar();

#ifndef SWIG
    operator double() const;
    operator SX() const;
    operator MX() const;
#endif

    double as_double() const { return this->operator double();}
    SX as_SX() const { return this->operator SX();}
    MX as_MX() const { return this->operator MX();}

    bool is_double() const;
    bool is_SX() const;
    bool is_MX() const;
    static std::vector<double> vector_double(const std::vector<AnyScalar>& v);
    static bool is_double(const std::vector<AnyScalar>& v);
    static bool is_SX(const std::vector<AnyScalar>& v);
    static bool is_MX(const std::vector<AnyScalar>& v);


  private:
    TensorType t;
    double data_double;
    SX data_sx;
    MX data_mx;
};

AnyScalar pow(const AnyScalar&x, int i);

class AnyTensor {
  public:
#ifndef SWIG
    AnyTensor& operator=(const AnyTensor&);
#endif
    //AnyTensor(const AnyScalar& s);
    AnyTensor(const AnyTensor& s);
    //AnyTensor(const AnyTensor&, const std::vector<int>& dim);
    AnyTensor(const DT & t);
    AnyTensor(const ST & t);
    AnyTensor(const MT & t);
    static AnyTensor unity();
    AnyTensor();
    bool is_double() const;
    bool is_SX() const;
    bool is_MX() const;
    std::vector<int> dims() const;
    //bool equals(const AnyTensor&rhs) const;

#ifndef SWIG
    operator DT() const;
    operator ST() const;
    operator MT() const;
#endif
    static AnyTensor vertcat(const std::vector<AnyScalar>& v);

    AnyTensor outer_product(const AnyTensor &b);
    AnyTensor inner(const AnyTensor&b);

  private:
    TensorType t;
    DT data_double;
    ST data_sx;
    MT data_mx;
};


AnyTensor vertcat(const std::vector<AnyScalar> & v);
AnyTensor vertcat(const std::vector<double> & v);

#endif
