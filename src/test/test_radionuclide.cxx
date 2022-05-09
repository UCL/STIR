//
//
/*
    Copyright (C) 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

/*!
  \file 
  \ingroup test
  \ingroup ancillary
  \brief A simple program to test stir::RadionuclideDB and stir::Radionuclide

  \author Kris Thielemans
*/
#include "stir/RunTests.h"
#include "stir/RadionuclideDB.h"
#include "stir/Radionuclide.h"

#include <iostream>


START_NAMESPACE_STIR

/*!
  \brief Class with tests for stir::RadionuclideDB and stir::Radionuclide
  \ingroup test
  \ingroup ancillary
*/
class RadionuclideTest : public RunTests
{
public:
  void run_tests();
};


void
RadionuclideTest::run_tests()
{
  const ImagingModality pt_mod(ImagingModality::PT);
  const ImagingModality nm_mod(ImagingModality::NM);

  std::cerr << "Constructing database\n";
  RadionuclideDB db;

  std::cerr << "Testing F-18 and Tc99-m\n";

  const auto F18_rnuclide = db.get_radionuclide(pt_mod, "^18^Fluorine");
  std::cerr << "F18 radionuclide: " << F18_rnuclide.parameter_info();
  check_if_equal(F18_rnuclide.get_half_life(), 6584.04F, "check on F18 half-life");
  check_if_equal(F18_rnuclide.get_energy(), 511.F, "check on F18 energy");
  check_if_equal(F18_rnuclide.get_branching_ratio(), 0.9686F, "check on F-18 branching ratio");
  check(F18_rnuclide.get_modality() == pt_mod, "check on F-18 modality");
    
  const auto Tc99m_rnuclide = db.get_radionuclide(nm_mod, "^99m^Technetium");
  std::cerr << "Tc-99m radionuclide: " << Tc99m_rnuclide.parameter_info();
  check_if_equal(Tc99m_rnuclide.get_half_life(), 21624.12, "check on Tc-99m half-life");
  check_if_equal(Tc99m_rnuclide.get_energy(), 140.511F, "check on Tc-99m energy");
  check_if_equal(Tc99m_rnuclide.get_branching_ratio(), 0.885F, "check on Tc-99m branching ratio");
  check(Tc99m_rnuclide.get_modality() == nm_mod, "check on Tc-99m modality");
  
  std::cerr << "Testing defaults\n";
  {
    // PET
    {
      const auto rnuclide = db.get_radionuclide(pt_mod, "");
      check(rnuclide == F18_rnuclide, "check \"empty\" radionuclide for PET is F-18");

      const auto def_rnuclide = db.get_radionuclide(pt_mod, "default");
      check(rnuclide == def_rnuclide, "check equality of \"empty\" and \"default\" radionuclide for PET");
    }
    // NM
    {
      const auto rnuclide = db.get_radionuclide(nm_mod, "");
      check(rnuclide == Tc99m_rnuclide, "check \"empty\" radionuclide for NM is Tc-99m");

      const auto def_rnuclide = db.get_radionuclide(nm_mod, "default");
      check(rnuclide == def_rnuclide, "check equality of \"empty\" and \"default\" radionuclide for NM");
    }
  }

#ifdef nlohmann_json_FOUND
  std::cerr << "Testing Yttrium-90\n";
  {
    const auto Y90_rnuclide_SPECT = db.get_radionuclide(nm_mod, "^90^Yttrium");
    std::cerr << "Y90 (SPECT) radionuclide: " << Y90_rnuclide_SPECT.parameter_info();
    check_if_equal(Y90_rnuclide_SPECT.get_half_life(), 230549.76, "check on Y90 (SPECT) half-life");
    check_if_equal(Y90_rnuclide_SPECT.get_energy(), 150.F, "check on Y90 (SPECT) energy");
    check_if_equal(Y90_rnuclide_SPECT.get_branching_ratio(), 0.99983F, "check on Y90 (SPECT) branching ratio");
    check(Y90_rnuclide_SPECT.get_modality() == nm_mod, "check on Y90 (SPECT) modality");

    const auto Y90_rnuclide_PET = db.get_radionuclide(pt_mod, "^90^Yttrium");
    std::cerr << "Y90 (PET) radionuclide: " << Y90_rnuclide_PET.parameter_info();
    check_if_equal(Y90_rnuclide_PET.get_half_life(), 230549.76, "check on Y90 (PET) half-life");
    check_if_equal(Y90_rnuclide_PET.get_energy(), 511.F, "check on Y90 (PET) energy");
    check_if_equal(Y90_rnuclide_PET.get_branching_ratio(), 0.0000319F, "check on Y90 (PET) branching ratio");
    check(Y90_rnuclide_PET.get_modality() == pt_mod, "check on Y90 (PET) modality");

  }
  
  std::cerr << "Testing lookup-table and database\n";
  {
    check(F18_rnuclide == db.get_radionuclide(pt_mod, "F-18"), "alias F-18");
    check(F18_rnuclide == db.get_radionuclide(pt_mod, "18F"), "alias 18F");
    check(Tc99m_rnuclide == db.get_radionuclide(nm_mod, "Tc-99m"), "alias Tc-99m");
    check(Tc99m_rnuclide == db.get_radionuclide(nm_mod, "99mTc"), "alias 99mTc");
    check_if_equal(db.get_radionuclide(pt_mod, "^11^Carbon").get_half_life(), 1221.66F,
                   "C11 half-life");
  }
#endif    

}


END_NAMESPACE_STIR



USING_NAMESPACE_STIR



int main()
{
  RadionuclideTest tests;
  tests.run_tests();
  return tests.main_return_value();
}
