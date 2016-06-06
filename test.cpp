#include <tensor.hpp>


int main(){

  ST t = ST::sym("t",{2,4,5});

  ST t2 = ST::sym("t2",{2,4,5});

  ST t3 = (t+t2)*t;

  std::cout << "t3" << t3.data_ << endl;

  DT t4 = DT(DM({{1,2,5,6},{3,4,7,8}}),{2,2,2});
  DT t5 = t4+t4;

  std::cout << "t" << t5.data_ << endl;

  return 0;
}
