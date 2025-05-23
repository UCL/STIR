/*
    Copyright (C) 2020-2021, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_test
  \ingroup OSMAPOSL
  \brief Test program for OSMAPOSL
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/test/PoissonLLReconstructionTests.h"
#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"

START_NAMESPACE_STIR

typedef DiscretisedDensity<3,float> target_type;
/*!
  \ingroup recon_test
  \ingroup OSMAPOSL
  \brief Test class for OSMAPOSL
*/
class TestOSMAPOSL : public PoissonLLReconstructionTests<target_type>
{
private:
  typedef PoissonLLReconstructionTests<target_type> base_type;
public:
  //! Constructor that can take some input data to run the test with
  TestOSMAPOSL(const std::string &projector_pair_filename = "",
               const std::string &proj_data_filename = "",
               const std::string & density_filename = "")
    : base_type(projector_pair_filename, proj_data_filename, density_filename)
  {}
  virtual ~TestOSMAPOSL() {}

  
  virtual void construct_reconstructor();
  OSMAPOSLReconstruction<target_type>&
  recon()
  { return dynamic_cast<OSMAPOSLReconstruction<target_type>& >(*this->_recon_sptr); }

  void run_tests();
};


void
TestOSMAPOSL::
construct_reconstructor()
{
  this->_recon_sptr.reset(new OSMAPOSLReconstruction<target_type>);
  this->construct_log_likelihood();
  this->recon().set_objective_function_sptr(this->_objective_function_sptr);
  //this->recon().set_num_subsets(4); // TODO should really check if this is appropriate for this->_proj_data_sptr->get_num_views()
  this->recon().set_num_subiterations(20);
}

void
TestOSMAPOSL::
run_tests()
{
  std::cerr << "Tests for OSMAPOSL\n";

  try {
    this->construct_input_data();
    this->construct_reconstructor();
    shared_ptr<target_type> output_sptr(this->_input_density_sptr->get_empty_copy());
    output_sptr->fill(1.F);
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

  if (everything_ok)
    {
      // see if it checks input parameters
      try
        {
          std::cerr << "\nYou should now see an error about a wrong setting for MAP model" << std::endl;
          this->recon().set_MAP_model("a_wrong_value");
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
        std::cerr << "\nUsage: " << argv[0] << " [projector_pair_filename [template_proj_data [image]]]\n"
                  << "projector_pair_filename (optional) can be used to specify the projectors\n"
                  <<"  if set to an empty string, the default ray-tracing matrix will be used.\n"
                  << "template_proj_data (optional) will serve as a template, but is otherwise not used.\n"
                  << "image (optional) has to be compatible with projection data and currently at zoom=1\n";
        return EXIT_FAILURE;
    }

    //set_default_num_threads();

    TestOSMAPOSL test(argc>1 ? argv[1] : "", argc > 2 ? argv[2] : "", argc > 3 ? argv[3] : "");

    if (test.is_everything_ok())
        test.run_tests();

    return test.main_return_value();
}
