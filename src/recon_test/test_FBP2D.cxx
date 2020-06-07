/*
    Copyright (C) 2020, University College London
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
  \ingroup recon_test
  \ingroup FBP2D
  \brief Test program for FBP2D
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/test/ReconstructionTests.h"
#include "stir/analytic/FBP2D/FBP2DReconstruction.h"

START_NAMESPACE_STIR

typedef DiscretisedDensity<3,float> target_type;
/*!
  \ingroup recon_test
  \ingroup FBP2D
  \brief Test class for FBP2D
*/
class TestFBP2D : public ReconstructionTests<target_type>
{
private:
  typedef ReconstructionTests<target_type> base_type;
public:
  //! Constructor that can take some input data to run the test with
  TestFBP2D(const std::string &proj_data_filename = "",
            const std::string & density_filename = "")
    : base_type(proj_data_filename, density_filename)
  {}
  virtual ~TestFBP2D() {}

  
  virtual void construct_reconstructor();
  void run_tests();
};


void
TestFBP2D::
construct_reconstructor()
{
  this->_recon_sptr.reset(new FBP2DReconstruction);
}

void
TestFBP2D::
run_tests()
{
  std::cerr << "Tests for FBP2D\n";

  try {
    this->construct_input_data();
    this->construct_reconstructor();
    shared_ptr<const target_type> output_sptr = this->reconstruct();
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

  // see if it checks input parameters
  {
    FBP2DReconstruction fbp(this->_proj_data_sptr, /*alpha*/ -1.F);
    try
      {
        std::cerr << "\nYou should now see an error about a wrong setting for alpha" << std::endl;
        fbp.set_up(this->_input_density_sptr);
        // we shouldn't get here
        everything_ok = false;
      }
    catch (...)
      {
      }
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

    TestFBP2D test(argc>1 ? argv[1] : "", argc > 2 ? argv[2] : "");

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}
