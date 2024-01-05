// Copyright © 2023 Apple Inc.

#include <set>

#include "doctest/doctest.h"
#include "mlx/mlx.h"

using namespace mlx::core;

TEST_CASE("einsum_path") {
  std::vector<std::tuple<
      std::vector<int>,
      std::set<char>,
      std::string,
      std::vector<std::string>,
      bool>>
      expected;
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'j'}, "jk,ij->ik", {"kl", "ik"}, true));
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'k'}, "ik,kl->il", {"il"}, true));
  auto x = einsum_path("ij,jk,kl", {ones({2, 2}), ones({2, 2}), ones({2, 2})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(std::get<0>(x.at(i)), std::get<0>(expected.at(i)));
    CHECK_EQ(std::get<1>(x.at(i)), std::get<1>(expected.at(i)));
    CHECK_EQ(std::get<2>(x.at(i)), std::get<2>(expected.at(i)));
    CHECK_EQ(std::get<3>(x.at(i)), std::get<3>(expected.at(i)));
    CHECK_EQ(std::get<4>(x.at(i)), std::get<4>(expected.at(i)));
  }
  expected.clear();
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'i'}, "i,i->", {""}, false));
  x = einsum_path("i,i", {ones({2}), ones({1})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(std::get<0>(x.at(i)), std::get<0>(expected.at(i)));
    CHECK_EQ(std::get<1>(x.at(i)), std::get<1>(expected.at(i)));
    CHECK_EQ(std::get<2>(x.at(i)), std::get<2>(expected.at(i)));
    CHECK_EQ(std::get<3>(x.at(i)), std::get<3>(expected.at(i)));
    CHECK_EQ(std::get<4>(x.at(i)), std::get<4>(expected.at(i)));
  }
  expected.clear();
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'j'}, "jk,ij->ik", {"ik"}, true));
  x = einsum_path("ij,jk", {ones({2, 2}), ones({2, 2})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(std::get<0>(x.at(i)), std::get<0>(expected.at(i)));
    CHECK_EQ(std::get<1>(x.at(i)), std::get<1>(expected.at(i)));
    CHECK_EQ(std::get<2>(x.at(i)), std::get<2>(expected.at(i)));
    CHECK_EQ(std::get<3>(x.at(i)), std::get<3>(expected.at(i)));
    CHECK_EQ(std::get<4>(x.at(i)), std::get<4>(expected.at(i)));
  }
  expected.clear();
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'i', 'j'}, "jil,ijk->kl", {"kl"}, true));

  x = einsum_path("ijk,jil->kl", {ones({3, 4, 5}), ones({4, 3, 2})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(std::get<0>(x.at(i)), std::get<0>(expected.at(i)));
    CHECK_EQ(std::get<1>(x.at(i)), std::get<1>(expected.at(i)));
    CHECK_EQ(std::get<2>(x.at(i)), std::get<2>(expected.at(i)));
    CHECK_EQ(std::get<3>(x.at(i)), std::get<3>(expected.at(i)));
    CHECK_EQ(std::get<4>(x.at(i)), std::get<4>(expected.at(i)));
  }

  expected.clear();
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>(
      {3, 0}, {'k'}, "nlk,ijk->injl", {"ilm", "njm", "abc", "injl"}, true));
  expected.emplace_back(
      std::make_tuple<
          std::vector<int>,
          std::set<char>,
          std::string,
          std::vector<std::string>,
          bool>({1, 0}, {'m'}, "njm,ilm->injl", {"abc", "injl", "injl"}, true));
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>(
      {2, 1}, {'n', 'i', 'l', 'j'}, "injl,injl->", {"abc", ""}, true));
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'c', 'a', 'b'}, ",abc->", {""}, false));

  x = einsum_path(
      "ijk,ilm,njm,nlk,abc->",
      {ones({2, 4, 8}),
       ones({2, 4, 8}),
       ones({2, 4, 8}),
       ones({2, 4, 8}),
       ones({2, 4, 8})});
  CHECK_EQ(x.size(), expected.size());
  for (int i = 0; i < x.size(); i++) {
    CHECK_EQ(std::get<0>(x.at(i)), std::get<0>(expected.at(i)));
    CHECK_EQ(std::get<1>(x.at(i)), std::get<1>(expected.at(i)));
    CHECK_EQ(std::get<2>(x.at(i)), std::get<2>(expected.at(i)));
    CHECK_EQ(std::get<3>(x.at(i)), std::get<3>(expected.at(i)));
    CHECK_EQ(std::get<4>(x.at(i)), std::get<4>(expected.at(i)));
  }

  expected.clear();
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>(
      {2, 0}, {'a'}, "abcd,ea->bcde", {"fb", "gc", "hd", "bcde"}, true));
  expected.emplace_back(
      std::make_tuple<
          std::vector<int>,
          std::set<char>,
          std::string,
          std::vector<std::string>,
          bool>({3, 0}, {'b'}, "bcde,fb->cdef", {"gc", "hd", "cdef"}, true));
  expected.emplace_back(
      std::make_tuple<
          std::vector<int>,
          std::set<char>,
          std::string,
          std::vector<std::string>,
          bool>({2, 0}, {'c'}, "cdef,gc->defg", {"hd", "defg"}, true));
  expected.emplace_back(std::make_tuple<
                        std::vector<int>,
                        std::set<char>,
                        std::string,
                        std::vector<std::string>,
                        bool>({1, 0}, {'d'}, "defg,hd->efgh", {"efgh"}, true));
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
    CHECK_EQ(std::get<0>(x.at(i)), std::get<0>(expected.at(i)));
    CHECK_EQ(std::get<1>(x.at(i)), std::get<1>(expected.at(i)));
    CHECK_EQ(std::get<2>(x.at(i)), std::get<2>(expected.at(i)));
    CHECK_EQ(std::get<3>(x.at(i)), std::get<3>(expected.at(i)));
    CHECK_EQ(std::get<4>(x.at(i)), std::get<4>(expected.at(i)));
  }
};