/*
    Copyright (C) 2011, Hammersmith Imanet Ltd
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup recon_test
  
  \brief Test program for stir::QuadraticPrior, RelativeDifferencePrior, and LogcoshPrior

  \par Usage

  <pre>
  test_priors [ density_filename ] 
  </pre>
  where the argument is optional. See the class documentation for more info.

  \author Kris Thielemans
  \author Robert Twyman
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/recon_buildblock/LogcoshPrior.h"
//#include "stir/recon_buildblock/PLSPrior.h"
#include "stir/RunTests.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include "stir/num_threads.h"
#include <iostream>
#include <memory>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

START_NAMESPACE_STIR


/*!
  \ingroup test
  \brief Test class for QuadraticPrior, RelativeDifferencePrior, and LogcoshPrior

  This test compares the result of GeneralisedPrior::compute_gradient()
  with a numerical gradient computed by using the 
  GeneralisedPrior::compute_value() function.

*/
class GeneralisedPriorTests : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  /*! This makes it possible to run the test with your own data. However, beware that
      it is very easy to set up a very long computation. 

      \todo it would be better to parse an objective function. That would allow us to set
      all parameters from the command line.
  */
  GeneralisedPriorTests(char const * const density_filename = 0);
  typedef DiscretisedDensity<3,float> target_type;
  void construct_input_data(shared_ptr<target_type>& density_sptr);

  void run_tests();
protected:
  char const * density_filename;
  shared_ptr<GeneralisedPrior<target_type> >  objective_function_sptr;

  //! run the test
  /*! Note that this function is not specific to a particular prior */
  void run_tests_for_objective_function(const std::string& test_name,
                                        GeneralisedPrior<target_type>& objective_function,
                                        shared_ptr<target_type> target_sptr);
};

GeneralisedPriorTests::
GeneralisedPriorTests(char const * const density_filename)
  : density_filename(density_filename)
{}

void
GeneralisedPriorTests::
run_tests_for_objective_function(const std::string& test_name,
                                 GeneralisedPrior<GeneralisedPriorTests::target_type>& objective_function,
                                 shared_ptr<GeneralisedPriorTests::target_type> target_sptr)
{
  std::cerr << "----- test " << test_name << '\n';
  if (!check(objective_function.set_up(target_sptr)==Succeeded::yes, "set-up of objective function"))
    return;

  // setup images
  target_type& target(*target_sptr);
  shared_ptr<target_type> gradient_sptr(target.get_empty_copy());
  shared_ptr<target_type> gradient_2_sptr(target.get_empty_copy());

  info("Computing gradient",3);
  objective_function.compute_gradient(*gradient_sptr, target);
  this->set_tolerance(std::max(fabs(double(gradient_sptr->find_min())), fabs(double(gradient_sptr->find_max()))) /1000);

  info("Computing objective function at target",3);
  const double value_at_target = objective_function.compute_value(target);
  target_type::full_iterator target_iter=target.begin_all();
  target_type::full_iterator gradient_iter=gradient_sptr->begin_all();
  target_type::full_iterator gradient_2_iter=gradient_2_sptr->begin_all(); 

  // setup perturbation response
  const float eps = 1e-3F;
  bool testOK = true;
  info("Computing gradient of objective function by numerical differences (this will take a while)",3);
  while(target_iter!=target.end_all())// && testOK)
    {
      const float org_image_value = *target_iter;
      *target_iter += eps;  // perturb current voxel
      const double value_at_inc = objective_function.compute_value(target);
      *target_iter = org_image_value; // restore
      const float ngradient_at_iter = static_cast<float>((value_at_inc - value_at_target)/eps);
      *gradient_2_iter = ngradient_at_iter;
      testOK = testOK && this->check_if_equal(ngradient_at_iter, *gradient_iter, "gradient");
      //for (int i=0; i<5 && target_iter!=target.end_all(); ++i)
        {
          ++gradient_2_iter; ++target_iter; ++ gradient_iter;
        }
    }
  if (!testOK)
    {
      info("Writing diagnostic files gradient" + test_name + ".hv, numerical_gradient" + test_name + ".hv");
      write_to_file("gradient" + test_name + ".hv", *gradient_sptr);
      write_to_file("numerical_gradient" + test_name + ".hv", *gradient_2_sptr);
    }

}

void
GeneralisedPriorTests::
construct_input_data(shared_ptr<target_type>& density_sptr)
{
  if (this->density_filename == 0)
    {
      // construct a small image with random voxel values between 0 and 1

      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
      exam_info_sptr->imaging_modality = ImagingModality::PT;
      CartesianCoordinate3D<float> origin (0,0,0);
      CartesianCoordinate3D<float> voxel_size(2.F,3.F,3.F);
      
      density_sptr.reset(new VoxelsOnCartesianGrid<float>(exam_info_sptr,
                                                          IndexRange<3>(make_coordinate(10,9,8)),
                                                          origin, voxel_size));
      // fill with random numbers between 0 and 1
      typedef boost::mt19937 base_generator_type;
      // initialize by reproducible seed
      static base_generator_type generator(boost::uint32_t(42));
      static boost::uniform_01<base_generator_type> random01(generator);
      for (target_type::full_iterator iter=density_sptr->begin_all(); iter!=density_sptr->end_all(); ++iter)
        *iter = static_cast<float>(random01());

    }
  else
    {
      // load image from file
      shared_ptr<target_type> aptr(read_from_file<target_type>(this->density_filename));
      density_sptr = aptr;
    }
    return;
}

void
GeneralisedPriorTests::
run_tests()
{
  shared_ptr<target_type> density_sptr;
  construct_input_data(density_sptr);

  std::cerr << "Tests for QuadraticPrior\n";
  {
    QuadraticPrior<float> objective_function(false, 1.F);
    this->run_tests_for_objective_function("Quadratic_no_kappa", objective_function, density_sptr);
  }
  std::cerr << "Tests for Relative Difference Prior\n";
  {
    // gamma is default and epsilon is off
    RelativeDifferencePrior<float> objective_function(false, 1.F, 2.F, 0.F);
    this->run_tests_for_objective_function("RDP_no_kappa", objective_function, density_sptr);
  }
  // Disabled PLS due to known issue
//  std::cerr << "Tests for PLSPrior\n";
//  {
//    PLSPrior<float> objective_function(false, 1.F);
//    shared_ptr<DiscretisedDensity<3,float> > anatomical_image_sptr(density_sptr->get_empty_copy());
//    anatomical_image_sptr->fill(1.F);
//    objective_function.set_anatomical_image_sptr(anatomical_image_sptr);
//    this->run_tests_for_objective_function("PLS_no_kappa_flat_anatomical", objective_function, density_sptr);
//  }
  std::cerr << "Tests for Logcosh Prior\n";
  {
    // scalar is off
    LogcoshPrior<float> objective_function(false, 1.F, 1.F);
    this->run_tests_for_objective_function("Logcosh_no_kappa", objective_function, density_sptr);
  }
} 

END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  set_default_num_threads();

  GeneralisedPriorTests tests(argc>1? argv[1] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
