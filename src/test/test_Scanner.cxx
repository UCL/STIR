//
// $Id$
//
/*!

  \file
  \ingroup test

  \brief Test program for Scanner hierarchy

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/

#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#ifdef HAVE_LLN_MATRIX
#include "ecat_model.h"
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
  cerr << "Tests for scanner model " << scanner.get_name()<<'\n';
  {
    const float natural_bin_size =
      scanner.get_ring_radius()*float(_PI)/scanner.get_num_detectors_per_ring();
    set_tolerance(.1);
    check_if_equal(natural_bin_size, scanner.get_default_bin_size(),
		   "comparing bin size derived from ring radius and num detectors with given");
    set_tolerance(.01);
  }
  {
    string name = scanner.get_name();
    name += " ";
    shared_ptr<Scanner> scanner_from_name_sptr =
      Scanner::get_scanner_from_name(name);
    check_if_equal(scanner.get_type(), scanner_from_name_sptr->get_type(),
		   "get_scanner_from_name");
  }
#ifdef HAVE_LLN_MATRIX
  {
    short ecat_type = ecat::find_ECAT_system_type(scanner);
    if (ecat_type==0)
      return;

    EcatModel * ecat_scanner_info = ecat_model(static_cast<int>(ecat_type));
    if (ecat_scanner_info==0)
      return;
    cerr << "Comparing STIR scanner info with LLN matrix\n";
    check_if_equal(scanner.get_num_axial_buckets(), ecat_scanner_info->rings,
		   "number of rings of buckets");
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
    check_if_equal(scanner.get_ring_radius(), ecat_scanner_info->crystalRad*10,
		   "detector radius");
    check_if_equal(scanner.get_ring_spacing(), ecat_scanner_info->planesep*5,
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
