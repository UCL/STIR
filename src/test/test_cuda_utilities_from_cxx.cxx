/*
    Copyright (C) 2026 University College London

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup test
  \ingroup CUDA

  \brief tests for the cuda_utilities in .cxx

  \author Kris Thielemans
*/

#include "stir/cuda_utilities.h"
#include "stir/Array.h"
#include "stir/RunTests.h"

#include <iostream>

START_NAMESPACE_STIR

/*!
  \brief Tests cuda_utilities functionality without CUDA
  \ingroup test
*/
class CUDACXXTests : public RunTests
{
public:
  void run_tests() override;
};

void
CUDACXXTests::run_tests()
{

  std::cerr << "Testing CUDA utilities in .cxx\n";

  CuVec<int> v(4, 5);
  check_if_equal(v.size(), std::size_t(4), "cuvec size");
  check_if_equal(v[1], 5, "cuvec element");

#if __cplusplus >= 202002L
  // std::allocate_shared only supports array types sice C++-20.
  auto sp = std::allocate_shared<int[]>(CuAlloc<int>(), 2);
  Array<1, int> a(2, sp);
  a.fill(5);
  check_if_equal(a.size(), std::size_t(2), "allocate_shared Array size");
  check_if_equal(a[1], 5, "allocate_shared Array element");
#endif
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  CUDACXXTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
