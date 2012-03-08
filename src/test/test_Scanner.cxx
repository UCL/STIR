//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup test

  \brief Test program for stir::Scanner hierarchy

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#ifdef HAVE_LLN_MATRIX
#include "ecat_model.h"
extern "C" {
  EcatModel *ecat_model(int);
}

#include "stir/IO/stir_ecat_common.h"
#endif
#include <iostream>
#include <math.h>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for Scanner
*/
class ScannerTests: public RunTests
{
public:  
  void run_tests();
private:
  void test_scanner(const Scanner&);
};


void
ScannerTests::
run_tests()
{
  Scanner::Type type= Scanner::E931; 
  while (type != Scanner::Unknown_scanner)
  {
    if (type!=Scanner::User_defined_scanner)
      test_scanner(Scanner(type));
    // tricky business to find next type
    int int_type = type;
    ++int_type;
    type = static_cast<Scanner::Type>(int_type);
  }
}

void
ScannerTests::
test_scanner(const Scanner& scanner)
{
  set_tolerance(.00001);
  cerr << "Tests for scanner model " << scanner.get_name()<<'\n';

  check(scanner.check_consistency() == Succeeded::yes, "check_consistency");

  /* check if number of non-arccorrected tangential positions is smaller than the maximum
     allowed for a full-ring tomograph 
  */
  {
    check(scanner.get_max_num_non_arccorrected_bins() <= scanner.get_num_detectors_per_ring(),
	  "too large max_num_non_arccorrected_bins compared to num_detectors_per_ring");
  }
  /* check if default_bin_size is close to the central bin size. This is especially true
     for CTI scanners.
     To avoid warnings/errors, we exclude some scanners where we know the bin-size for sure
     from the test.
  */
  if (scanner.get_type() != Scanner::Advance &&
      scanner.get_type() != Scanner::DiscoveryLS &&
      scanner.get_type() != Scanner::DiscoveryST &&
      scanner.get_type() != Scanner::DiscoverySTE &&
      scanner.get_type() != Scanner::DiscoveryRX/* &&
      scanner.get_type() != Scanner::HZLR*/)
  {
    const float natural_bin_size =
      scanner.get_inner_ring_radius()*float(_PI)/scanner.get_num_detectors_per_ring();
    if (fabs(natural_bin_size - scanner.get_default_bin_size())> .03)
      warning("central bin size (derived from inner ring radius and num detectors) %g\n"
	      "differs from given default bin size %g\n"
	      "(unequal values do not necessarily mean there's an error as "
	      "it's a convention used by the scanner manufacturer)\n",
	      natural_bin_size, scanner.get_default_bin_size());
  }
  // (weak) test on get_scanner_from_name
  {
    string name = scanner.get_name();
    name += " ";
    shared_ptr<Scanner> scanner_from_name_sptr(Scanner::get_scanner_from_name(name));
    check_if_equal(scanner.get_type(), scanner_from_name_sptr->get_type(),
		   "get_scanner_from_name");
  }
#ifdef HAVE_LLN_MATRIX
  if (scanner.get_type() <= Scanner::E966) // TODO relies on ordering of enum
  {
    // compare with info from ecat_model

    cerr << "Comparing STIR scanner info with LLN matrix\n";
    short ecat_type = ecat::find_ECAT_system_type(scanner);
    if (ecat_type==0)
      return;

    EcatModel * ecat_scanner_info = ecat_model(static_cast<int>(ecat_type));
    if (ecat_scanner_info==0)
      return;
    check_if_equal(scanner.get_num_axial_buckets(), ecat_scanner_info->rings,
		   "number of rings of buckets");
    if (scanner.get_type() != Scanner::E925) // ART is a partial ring tomograph
      check_if_equal(scanner.get_num_axial_buckets()*scanner.get_num_transaxial_buckets(), 
		   ecat_scanner_info->nbuckets,
		   "total number of buckets");
    check_if_equal(scanner.get_num_transaxial_blocks_per_bucket(), ecat_scanner_info->transBlocksPerBucket,
		   "transaxial blocks per bucket");
    check_if_equal(scanner.get_num_axial_blocks_per_bucket(), ecat_scanner_info->axialBlocksPerBucket,
		   "axial blocks per bucket");
    check_if_equal(scanner.get_num_transaxial_blocks_per_bucket() * scanner.get_num_axial_blocks_per_bucket(), 
		   ecat_scanner_info->blocks,
		   "total number of blocks");
    check_if_equal(scanner.get_num_axial_crystals_per_block(), ecat_scanner_info->axialCrystalsPerBlock,
		   "number of crystals in the axial direction");
    check_if_equal(scanner.get_num_transaxial_crystals_per_block(), ecat_scanner_info->angularCrystalsPerBlock,
		   "number of transaxial crystals");
    check_if_equal(scanner.get_inner_ring_radius(), ecat_scanner_info->crystalRad*10,
		   "detector radius");
    check_if_equal(scanner.get_ring_spacing()/2, ecat_scanner_info->planesep*10,
		   "plane separation");
    check_if_equal(scanner.get_default_bin_size(), ecat_scanner_info->binsize*10,
		   "bin size (spacing of transaxial elements)");
  }
#endif
}  

END_NAMESPACE_STIR


int main()
{
USING_NAMESPACE_STIR

  ScannerTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
