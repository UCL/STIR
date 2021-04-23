/*
    Copyright (C) 2020, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_test
  \ingroup FBP3DRP
  \brief Test program for FBP3DRP
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/test/ReconstructionTests.h"
#include "stir/analytic/FBP3DRP/FBP3DRPReconstruction.h"

START_NAMESPACE_STIR

typedef DiscretisedDensity<3,float> target_type;
/*!
  \ingroup recon_test
  \ingroup FBP3DRP
  \brief Test class for FBP3DRP
*/
class TestFBP3DRP : public ReconstructionTests<target_type>
{
private:
  typedef ReconstructionTests<target_type> base_type;
public:
  //! Constructor that can take some input data to run the test with
  TestFBP3DRP(const std::string &proj_data_filename = "",
            const std::string & density_filename = "")
    : base_type(proj_data_filename, density_filename)
  {}
  virtual ~TestFBP3DRP() {}

  
  virtual void construct_reconstructor();
  void run_tests();
};


void
TestFBP3DRP::
construct_reconstructor()
{
  this->_recon_sptr.reset(new FBP3DRPReconstruction);
}

void
TestFBP3DRP::
run_tests()
{
  std::cerr << "Tests for FBP3DRP\n";

  try {
    this->construct_input_data();
    this->construct_reconstructor();
    shared_ptr<target_type> output_sptr(this->_input_density_sptr->get_empty_copy());
    this->reconstruct(output_sptr);
    this->compare(output_sptr);
  }
  catch(const std::exception &error)
    {
      std::cerr << "\nHere's the error:\n\t" << error.what() << "\n\n";
      everything_ok = false;
    }
  catch(...)
    {
      everything_ok = false;
    }
}

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  if (argc < 1 || argc > 3) {
    std::cerr << "\n\tUsage: " << argv[0] << " [template_proj_data [image]]\n"
              << "template_proj_data (optional) will serve as a template, but is otherwise not used.\n"
              << "Image (optional) has to be compatible with projection data and currently at zoom=1\n";
    return EXIT_FAILURE;
  }

  //set_default_num_threads();

  TestFBP3DRP test(argc>1 ? argv[1] : "", argc > 2 ? argv[2] : "");

  if (test.is_everything_ok())
    test.run_tests();

  return test.main_return_value();
}
