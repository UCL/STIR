
/*
    Copyright (C) 2025, UMCG
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup test

  \brief Test program for stir::PETSIRD hierarchy

  \author Nikos Efthimiou

*/
#include "stir/PETSIRDInfo.h"
#include "stir/RunTests.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

class PETSIRDTests : public RunTests
{
public:
  void run_tests() override;

private:
  shared_ptr<PETSIRDInfo> petsird_info_sptr;
};

void
PETSIRDTests::run_tests()
{}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  PETSIRDTests test;
  test.run_tests();
  return test.main_return_value();
}
