#include <tensor.hpp>


int main(){

  ST t = ST::sym("t",{2,4,5});

  ST t2 = ST::sym("t2",{2,4,5});

  ST t3 = (t+t2)*t;

  std::cout << "t3" << t3.data_ << endl;

  DT t4 = DT(DM({{1,2,5,6},{3,4,7,8}}),{2,2,2});
  DT t5 = t4+t4;

  std::cout << "t" << t5.data_ << endl;

  // test sub2ind, ind2sub
  std::vector<int> dims = {3,2,4};
  for (int i=0;i<3*2*4;i++) {
    std::vector<int> ind = Tensor<DM>::sub2ind(dims,i);
    int j = Tensor<DM>::ind2sub(dims,ind);
    assert(i==j);
  }

  DM expected1 = DM({{2,10,4,12},{6,14,8,16}});
  DM got1 = t5.reorder_dims({0,2,1}).data_;

  assert(double(norm_inf(got1-expected1))==0);

  DM expected2 = DM({{2,10,6,14},{4,12,8,16}});
  DM got2 = t5.reorder_dims({1,2,0}).data_;
  assert(double(norm_inf(got2-expected2))==0);


  return 0;
}
