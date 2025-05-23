/*
    Copyright (C) 2020, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
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

  virtual std::unique_ptr<ProjDataInfo>
    construct_default_proj_data_info_uptr() const;
  
  virtual void construct_reconstructor();
  void run_tests();
};

std::unique_ptr<ProjDataInfo>
TestFBP2D::
construct_default_proj_data_info_uptr() const
{
  // construct a small scanner and sinogram
  shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
  // currently need this for limitation in the backprojector
  scanner_sptr->set_intrinsic_azimuthal_tilt(0.F);
  scanner_sptr->set_num_rings(5);
  std::unique_ptr<ProjDataInfo> proj_data_info_uptr(
        ProjDataInfo::ProjDataInfoCTI(scanner_sptr,
                                      /*span=*/3,
                                      /*max_delta=*/4,
                                      /*num_views=*/128,
                                      /*num_tang_poss=*/128));
  return proj_data_info_uptr;
}

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
