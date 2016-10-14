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
    explicit operator double() const;
    explicit operator SX() const;
    explicit operator MX() const;
#endif

    double as_double() const { return this->operator double();}
    SX as_SX() const { return this->operator SX();}
    MX as_MX() const { return this->operator MX();}

    bool is_double() const;
    bool is_SX() const;
    bool is_MX() const;

    static std::vector<double> as_double(const std::vector<AnyScalar>& v);
    static std::vector<SX> as_SX(const std::vector<AnyScalar>& v);
    static std::vector<MX> as_MX(const std::vector<AnyScalar>& v);
    static TensorType merge(TensorType a, TensorType b);

#define ANYSCALAR_BINARY(OP) \
switch (AnyScalar::merge(x.t, y.t)) { \
  case TENSOR_DOUBLE: \
    return x.as_double() OP y.as_double();break; \
  case TENSOR_SX: \
    return x.as_SX() OP y.as_SX();break; \
  case TENSOR_MX: \
    return x.as_MX() OP y.as_MX();break; \
  default:\
     assert(false); return 0;\
}

    /// Logic greater or equal to
    friend inline AnyScalar operator>=(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(>=)
    }

    friend inline AnyScalar operator<(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(<)
    }
    friend inline AnyScalar operator>(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(>)
    }
    friend inline AnyScalar operator<=(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(<=)
    }
    friend inline AnyScalar operator-(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(-)
    }
    friend inline AnyScalar operator*(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(*)
    }
    friend inline AnyScalar operator/(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(/)
    }
    friend inline AnyScalar operator==(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(==)
    }
    friend inline AnyScalar operator&&(const AnyScalar &x, const AnyScalar &y) {
      ANYSCALAR_BINARY(&&)
    }

    AnyScalar& operator+=(const AnyScalar& rhs);

    static TensorType type(const std::vector<AnyScalar>& v);
    static bool is_double(const std::vector<AnyScalar>& v) {return type(v)==TENSOR_DOUBLE;}
    static bool is_SX(const std::vector<AnyScalar>& v) {return type(v)==TENSOR_SX;}
    static bool is_MX(const std::vector<AnyScalar>& v) {return type(v)==TENSOR_MX;}

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
    bool is_DT() const;
    bool is_ST() const;
    bool is_MT() const;
    DT as_DT() const { return this->operator DT();}
    ST as_ST() const { return this->operator ST();}
    MT as_MT() const { return this->operator MT();}
    static std::vector<DT> as_DT(const std::vector<AnyTensor>& v);
    static std::vector<ST> as_ST(const std::vector<AnyTensor>& v);
    static std::vector<MT> as_MT(const std::vector<AnyTensor>& v);
    static bool is_DT(const std::vector<AnyTensor>& v) {return type(v)==TENSOR_DOUBLE;}
    static bool is_ST(const std::vector<AnyTensor>& v) {return type(v)==TENSOR_SX;}
    static bool is_MT(const std::vector<AnyTensor>& v) {return type(v)==TENSOR_MX;}

    std::vector<int> dims() const;
    //bool equals(const AnyTensor&rhs) const;

    static TensorType type(const std::vector<AnyTensor>& v);

    static AnyTensor solve(const AnyTensor& A, const AnyTensor& B);

#ifndef SWIG
    explicit operator DT() const;
    explicit operator ST() const;
    explicit operator MT() const;
#endif
    static AnyTensor vertcat(const std::vector<AnyScalar>& v);
    static AnyTensor concat(const std::vector<AnyTensor>& v, int axis);

    static AnyTensor pack(const std::vector<AnyTensor>& v, int axis);
    static std::vector<AnyTensor> unpack(const AnyTensor& v, int axis);

    AnyTensor reorder_dims(const std::vector<int>& order) const;
    AnyTensor shape(const std::vector<int>& dims) const;

    AnyTensor outer_product(const AnyTensor &b);
    AnyTensor inner(const AnyTensor&b);

    #ifndef SWIG
    /// Print a representation of the object to a stream (shorthand)
    inline friend
        std::ostream& operator<<(std::ostream &stream, const AnyTensor& obj) {
            return stream << obj.getRepresentation();
        }
    #endif // SWIG

    std::string getRepresentation() const {
      if (is_DT()) return "AnyTensor:" + as_DT().getRepresentation();
      if (is_ST()) return "AnyTensor:" + as_ST().getRepresentation();
      if (is_MT()) return "AnyTensor:" + as_MT().getRepresentation();
      return "";
    }


  private:
    TensorType t;
    DT data_double;
    ST data_sx;
    MT data_mx;
};


AnyTensor vertcat(const std::vector<AnyScalar> & v);
AnyTensor vertcat(const std::vector<double> & v);

#endif
