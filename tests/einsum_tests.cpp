// Copyright © 2023 Apple Inc.

#include <set>

#include "doctest/doctest.h"
#include "mlx/einsum.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("test einsum path") {
  /*  std::vector<EinsumPath> expected;
    expected.push_back({{1, 0}, {'j'}, "jk,ij->ik", true});
    expected.push_back({{1, 0}, {'k'}, "ik,kl->il", true});
    auto x = einsum_path("ij,jk,kl", {ones({2, 2}), ones({2, 2}), ones({2,
    2})}); CHECK_EQ(x.size(), expected.size()); for (int i = 0; i < x.size();
    i++) { CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }
    expected.clear();
    expected.push_back(
        {{
             0,
         },
         {},
         "jki->ijk",
         false});
    x = einsum_path("jki", {ones({2, 3, 4})});
    CHECK_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); i++) {
      CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }
    expected.clear();
    expected.push_back({{1, 0}, {'i'}, "i,i->", false});
    x = einsum_path("i,i", {ones({2}), ones({1})});
    CHECK_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); i++) {
      CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }
    expected.clear();
    expected.push_back({{1, 0}, {'j'}, "jk,ij->ik", true});
    x = einsum_path("ij,jk", {ones({2, 2}), ones({2, 2})});
    CHECK_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); i++) {
      CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }
    expected.clear();
    expected.push_back({{1, 0}, {'i', 'j'}, "jil,ijk->kl", true});

    x = einsum_path("ijk,jil->kl", {ones({3, 4, 5}), ones({4, 3, 2})});
    CHECK_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); i++) {
      CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }
    expected.clear();
    expected.push_back({{3, 0}, {'k'}, "nlk,ijk->injl", true});
    expected.push_back({{1, 0}, {'m'}, "njm,ilm->injl", true});
    expected.push_back({{2, 1}, {'n', 'i', 'l', 'j'}, "injl,injl->", true});
    expected.push_back({{1, 0}, {'c', 'a', 'b'}, ",abc->", false});

    x = einsum_path(
        "ijk,ilm,njm,nlk,abc->",
        {ones({2, 4, 8}),
         ones({2, 4, 8}),
         ones({2, 4, 8}),
         ones({2, 4, 8}),
         ones({2, 4, 8})});
    CHECK_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); i++) {
      CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }
    expected.clear();
    expected.push_back({{2, 0}, {'a'}, "abcd,ea->bcde", true});
    expected.push_back({{3, 0}, {'b'}, "bcde,fb->cdef", true});
    expected.push_back({{2, 0}, {'c'}, "cdef,gc->defg", true});
    expected.push_back({{1, 0}, {'d'}, "defg,hd->efgh", true});
    x = einsum_path(
        "ea,fb,abcd,gc,hd->efgh",
        {
            ones({10, 10}),
            ones({10, 10}),
            ones({10, 10, 10, 10}),
            ones({10, 10}),
            ones({10, 10}),
        });
    CHECK_EQ(x.size(), expected.size());
    for (int i = 0; i < x.size(); i++) {
      CHECK_EQ(x.at(i).args, expected.at(i).args);
      CHECK_EQ(x.at(i).einsum_str, expected.at(i).einsum_str);
      CHECK_EQ(x.at(i).removing, expected.at(i).removing);
      CHECK_EQ(x.at(i).can_dot, expected.at(i).can_dot);
    }*/
}

TEST_CASE("test einsum") {
  CHECK_THROWS(einsum("i,j", {array({1.0})}));
  CHECK_THROWS(einsum("ijk", {full({2, 2}, 2.0f)}));
  CHECK_THROWS(einsum("", {}));
  CHECK_THROWS(einsum("ij", {array({1, 2})}));
  CHECK_THROWS(einsum("", {array({1, 2})}));
  CHECK_THROWS(einsum("i,ij", {array({1, 2}), array({2, 3})}));
  CHECK_THROWS(einsum("i,i", {array({1, 2}), array({2, 3, 4})}));
  CHECK_THROWS(einsum("i->ii", {array({1, 2})}));
  CHECK_THROWS(einsum("12", {zeros({4, 4})}));
  CHECK_THROWS(einsum("ii->i", {zeros({3, 2})}));
  //  CHECK_THROWS(einsum("ii->i", {zeros({3, 1})}));

  /*  auto x = einsum("jki", {full({2, 3, 4}, 3.0f)});
    CHECK_EQ(x.shape(), std::vector<int>{4, 2, 3});
    CHECK_EQ(x.dtype(), float32);
    auto expected = full({4, 2, 3}, 3.0f);
    CHECK_EQ(array_equal(x, expected).item<bool>(), true);
    x = einsum("ij,jk->ik", {full({2, 2}, 2.0f), full({2, 2}, 3.0f)});
    CHECK_EQ(x.shape(), std::vector<int>{2, 2});
    CHECK_EQ(x.dtype(), float32);
    expected = array({12.0f, 12.0f, 12.0f, 12.0f}, {2, 2});
    CHECK_EQ(array_equal(x, expected).item<bool>(), true);
    x = einsum("i,j->ij", {full({10}, 15.0f), full({10}, 20.0f)});
    CHECK_EQ(x.shape(), std::vector<int>{10, 10});
    CHECK_EQ(x.dtype(), float32);
    expected = full({10, 10}, 300.0f);
    CHECK_EQ(array_equal(x, expected).item<bool>(), true);
    x = einsum(
        "ijkl ,mlopq ->ikmop",
        {full({4, 5, 9, 4}, 20.0f), full({14, 4, 16, 7, 5}, 10.0f)});
    CHECK_EQ(x.shape(), std::vector<int>{4, 9, 14, 16, 7});
    CHECK_EQ(x.dtype(), float32);
    expected = full({4, 9, 14, 16, 7}, 20000.0f);
    CHECK_EQ(x.shape(), expected.shape());
    CHECK_EQ(array_equal(x, expected).item<bool>(), true);*/
}
