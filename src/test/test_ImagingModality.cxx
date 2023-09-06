/*!

  \file
  \ingroup test

  \brief Test program for stir::ImagingModality

  \author Kris Thielemans


*/
/*
    Copyright (C) 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ImagingModality.h"
#include "stir/RunTests.h"

#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR


/*!
  \brief Test class for ImagingModality
  \ingroup buildblock
  \ingroup test
*/
class ImagingModalityTests : public RunTests
{
public:
  void run_tests();
};

void
ImagingModalityTests::run_tests()
{
  cerr << "Tests for ImagingModality\n"
       << "Everythings is fine if the program runs without any output." << endl;

  {
    ImagingModality mod("PET");
    check(mod.get_modality() == ImagingModality::PT, "construct from string, enum PT");
    check_if_equal(mod.get_name(), "PT", "construct from string, string PT");
    ImagingModality mod2(ImagingModality::PT);
    check(mod == mod2, "equality");
  }
  {
    ImagingModality mod("nucMed");
    check(mod.get_modality() == ImagingModality::NM, "construct from string, enum NM");
    check_if_equal(mod.get_name(), "NM", "construct from string, string NM");
    ImagingModality mod2(ImagingModality::NM);
    check(mod == mod2, "equality");
  }
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main()
{
  ImagingModalityTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
