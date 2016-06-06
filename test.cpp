#include <tensor.hpp>


int main() {

  ST t = ST::sym("t", {2, 4, 5});

  ST t2 = ST::sym("t2", {2, 4 , 5});

  ST t3 = (t+t2)*t;

  std::cout << "t3" << t3.data() << endl;

  DT t4 = DT(DM({{1, 2, 5, 6}, {3, 4, 7, 8}}), {2, 2, 2});
  DT t5 = t4+t4;

  std::cout << "t" << t5.data() << endl;

  // test sub2ind, ind2sub
  std::vector<int> dims = {3, 2, 4};
  for (int i=0;i<3*2*4;i++) {
    std::vector<int> ind = Tensor<DM>::sub2ind(dims, i);
    int j = Tensor<DM>::ind2sub(dims, ind);
    assert(i==j);
  }

  DM expected = DM({{2, 10, 4, 12}, {6, 14, 8, 16}});
  DM got = t5.reorder_dims({0, 2, 1}).data();


  assert(static_cast<double>(norm_inf(got-expected))==0);

  expected = DM({{2, 10, 6, 14}, {4, 12, 8, 16}});
  got = t5.reorder_dims({1, 2, 0}).data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  DT t6 = DT(DM(std::vector<std::vector<double> >{{3, 4}, {1, 7}}), {2, 2});

  DT t7 = t6.partial_product(t5);

  expected = DM({{30, 44, 86, 100}, {44, 60, 108, 124}});
  got = t7.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  DT t8 = t7.partial_product(t5);
  expected = DM({{324, 472, 2260, 2632}, {448, 656, 2816, 3280}});
  got = t8.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  std::cout << t6.outer_product(t5).data() << std::endl;

  DT t10 = t5.slice({0, -1, -1});
  expected = DM(std::vector<std::vector<double> >{{2, 10}, {4, 12}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  return 0;
}
