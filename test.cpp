#include <tensor.hpp>


template<class T, class S>
void assert_equal(T a, S b) {
  assert(static_cast<double>(norm_inf(a-b))==0);
}


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

  // Outer product
  t8 = DT(DM(std::vector<std::vector<double> >{{3, 4}, {1, 7}}), {2, 2});
  DT t9 = DT(DM(std::vector<double>{1, 2}), {2, 1});

  DT t10 = t8.outer_product(t9);
  assert((t10.dims()==std::vector<int>{2, 2, 2, 1}));

  expected = DM({{3, 4, 6, 8}, {1, 7, 2, 14}});
  got = t10.data();


  assert(static_cast<double>(norm_inf(got-expected))==0);

  t9 = DT(DM(std::vector<double>{1, 2}), {1, 2});

  t10 = t8.outer_product(t9);
  assert((t10.dims()==std::vector<int>{2, 2, 1, 2}));

  expected = DM({{3, 4, 6, 8}, {1, 7, 2, 14}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t9 = DT(DM(std::vector<double>{1, 2}), {2});

  t10 = t8.outer_product(t9);
  assert((t10.dims()==std::vector<int>{2, 2, 2}));

  expected = DM({{3, 4, 6, 8}, {1, 7, 2, 14}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  // Slice tests
  t10 = t5.slice({0, -1, -1});
  expected = DM(std::vector<std::vector<double> >{{2, 10}, {4, 12}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t10 = t5.slice({1, -1, -1});
  expected = DM(std::vector<std::vector<double> >{{6, 14}, {8, 16}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t10 = t5.slice({-1, 0, -1});
  expected = DM(std::vector<std::vector<double> >{{2, 10}, {6, 14}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t10 = t5.slice({-1, 1, -1});
  expected = DM(std::vector<std::vector<double> >{{4, 12}, {8, 16}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t10 = t5.slice({-1, -1, 0});
  expected = DM(std::vector<std::vector<double> >{{2, 4}, {6, 8}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t10 = t5.slice({-1, -1, 1});
  expected = DM(std::vector<std::vector<double> >{{10, 12}, {14, 16}});
  got = t10.data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  t8 = DT(DM(std::vector<std::vector<double> >{{3, 4}, {1, 7}}), {2, 2});
  t9 = DT(DM(std::vector<double>{1, 2}), {2, 1});

  t8.einstein(t9, {-1, -2}, {-2, -3}, {-1, -3});

  expected = DM(3);
  got = t8.slice({0, 0}).data();
  assert(static_cast<double>(norm_inf(got-expected))==0);

  expected = DM(4);
  got = t8.slice({0, 1}).data();
  assert(static_cast<double>(norm_inf(got-expected))==0);

  // Inner product
  DT v1 = DT(DM(std::vector<double>{0, 1, 2}), {3});
  DT v2 = DT(DM(std::vector<double>{1, 2, 3}), {3});


  DT m1 = DT(DM(std::vector<std::vector<double>>{{0, 1}, {2, 3}, {4, 5}}), {3, 2});

  expected = DM(std::vector<double>{10, 13});
  got = v1.einstein(m1, {-1}, {-1,-2}, {-2}).data();

  assert(static_cast<double>(norm_inf(got-expected))==0);

  return 0;
}
