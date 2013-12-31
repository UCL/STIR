/*
    Copyright (C) 2011, Hammersmith Imanet
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
  
  \brief Test program for stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData
    
  \author Kris Thielemans
*/

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInMemory.h"
#include "stir/SegmentByView.h"
#include "stir/Scanner.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
//#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/RunTests.h"
#include "stir/IO/read_from_file.h"
#include "stir/info.h"
#include "stir/Succeeded.h"
#include <iostream>
#include <memory>
#include <boost/random/uniform_01.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>

#include "stir/IO/OutputFileFormat.h"
#include "stir/recon_buildblock/distributable_main.h"
START_NAMESPACE_STIR


/*!
  \ingroup test
  \brief Test class for PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  This is a somewhat preliminary implementation of a test that compares the result
  of GeneralisedObjectiveFunction::compute_gradient 
  with a numerical gradient computed by using the 
  GeneralisedObjectiveFunction::compute_objective_function() function.

  The trouble with this is that compute the gradient voxel by voxel is obviously 
  terribly slow. A solution (for the test) would be to compute it only in 
  a subset of voxels or so. We'll leave this for later.

  Note that the test only works if the objective function is well-defined. For example,
  if certain projections are non-zero, while the model estimates them to be zero, the
  Poisson objective function is in theory infinite.
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData uses some thresholds to try to 
  avoid overflow, but if there are too many of these bins, the total objective
  function will become infinite. The numerical gradient then becomes ill-defined
  (even in voxels that do not contribute to these bins).

*/
class PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests : public RunTests
{
public:
  //! Constructor that can take some input data to run the test with
  /*! This makes it possible to run the test with your own data. However, beware that
      it is very easy to set up a very long computation. See also the note about
      non-zero measured bins.

      \todo it would be better to parse an objective function. That would allow us to set
      all parameters from the command line.
  */
  PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests(char const * const proj_data_filename = 0, char const * const density_filename = 0);
  typedef DiscretisedDensity<3,float> target_type;
  void construct_input_data(shared_ptr<target_type>& density_sptr);

  void run_tests();
protected:
  char const * proj_data_filename;
  char const * density_filename;
  shared_ptr<GeneralisedObjectiveFunction<target_type> >  objective_function_sptr;

  //! run the test
  /*! Note that this function is not specific to PoissonLogLikelihoodWithLinearModelForMeanAndProjData */
  void run_tests_for_objective_function(GeneralisedObjectiveFunction<target_type>& objective_function,
                                        target_type& target);
};

PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests(char const * proj_data_filename, char const * const density_filename)
  : proj_data_filename(proj_data_filename), density_filename(density_filename)
{}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::
run_tests_for_objective_function(GeneralisedObjectiveFunction<PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::target_type>& objective_function,
                                 PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::target_type& target) {
  shared_ptr<target_type> gradient_sptr(target.get_empty_copy());
  shared_ptr<target_type> gradient_2_sptr(target.get_empty_copy());
  const int subset_num = 0;
  info("Computing gradient");
  objective_function.compute_sub_gradient(*gradient_sptr, target, subset_num);
  this->set_tolerance(max(fabs(double(gradient_sptr->find_min())), double(gradient_sptr->find_max()))/1000);
  info("Computing objective function at target");
  const double value_at_target = objective_function.compute_objective_function(target, subset_num);
  target_type::full_iterator target_iter=target.begin_all();
  target_type::full_iterator gradient_iter=gradient_sptr->begin_all();
  target_type::full_iterator gradient_2_iter=gradient_2_sptr->begin_all(); 
  const float eps = 1e-2F;
  bool testOK = true;
  info("Computing gradient of objective function by numerical differences (this will take a while)");
  while(target_iter!=target.end_all())
    {
      *target_iter += eps;
      const double value_at_inc = objective_function.compute_objective_function(target, subset_num);
      *target_iter -= eps;
      const float gradient_at_iter = static_cast<float>((value_at_inc - value_at_target)/eps);
      *gradient_2_iter++ = gradient_at_iter;
      testOK = testOK && 
        this->check_if_equal(gradient_at_iter, *gradient_iter, "gradient");
      ++target_iter; ++ gradient_iter;
    }
  if (~testOK)
    {
      info("Writing diagnostic files gradient.hv, numerical_gradient.hv");
      OutputFileFormat<target_type>::default_sptr()->write_to_file("gradient.hv", *gradient_sptr);
      OutputFileFormat<target_type>::default_sptr()->write_to_file("numerical_gradient.hv", *gradient_2_sptr);
#if 0
      OutputFileFormat<target_type>::default_sptr()->write_to_file("subsens.hv", 
                                                                   reinterpret_cast<const PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type> &>(objective_function).get_subset_sensitivity(subset_num));
      gradient_sptr->fill(0.F);
      reinterpret_cast<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type> &>(objective_function).
        compute_sub_gradient_without_penalty_plus_sensitivity(*gradient_sptr, target, subset_num);
      OutputFileFormat<target_type>::default_sptr()->write_to_file("gradient-without-sens.hv", *gradient_sptr);
#endif
    }

}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::
construct_input_data(shared_ptr<target_type>& density_sptr)
{ 
  shared_ptr<ProjData> proj_data_sptr;
  if (this->proj_data_filename == 0)
    {
      // construct a small scanner and sinogram
      shared_ptr<Scanner> scanner_sptr(new Scanner(Scanner::E953));
      scanner_sptr->set_num_rings(5);
      shared_ptr<ProjDataInfo> proj_data_info_sptr(
        ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
                                      /*span=*/3, 
                                      /*max_delta=*/4,
                                      /*num_views=*/16,
                                      /*num_tang_poss=*/16));
      shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
      proj_data_sptr.reset(new ProjDataInMemory (exam_info_sptr, proj_data_info_sptr));
      for (int seg_num=proj_data_sptr->get_min_segment_num(); 
           seg_num<=proj_data_sptr->get_max_segment_num();
           ++seg_num)
        {
          SegmentByView<float> segment = proj_data_sptr->get_empty_segment_by_view(seg_num);
          // fill in some crazy values
          float value=0;
          for (SegmentByView<float>::full_iterator iter = segment.begin_all();
               iter != segment.end_all();
               ++iter)
            {
              value = float(fabs((seg_num+.1)*value - 5)); // needs to be positive for Poisson
              *iter = value;
            }
          proj_data_sptr->set_segment(segment);
        }
    }
  else
    {
      proj_data_sptr =
        ProjData::read_from_file(this->proj_data_filename);
    }

  if (this->density_filename == 0)
    {
      // construct a small image

      CartesianCoordinate3D<float> origin (0,0,0);    
      const float zoom=1.F;
      
      density_sptr.reset(new VoxelsOnCartesianGrid<float>(*proj_data_sptr->get_proj_data_info_ptr(),zoom,origin));
      // fill with random numbers between 0 and 1
      typedef boost::mt19937 base_generator_type;
      // initialize by reproducible seed
      static base_generator_type generator(boost::uint32_t(42));
      static boost::uniform_01<base_generator_type> random01(generator);
      for (target_type::full_iterator iter=density_sptr->begin_all(); iter!=density_sptr->end_all(); ++iter)
        *iter = random01();

    }
  else
    {
      std::auto_ptr<target_type> aptr(read_from_file<target_type>(this->density_filename));
      density_sptr = aptr;
    }

  // make odd to avoid difficulties with outer-bin that isn't filled-in when using symmetries
  {
     BasicCoordinate<3,int> min_ind, max_ind;
      if (density_sptr->get_regular_range(min_ind,max_ind))
        {
          for (int d=2; d<=3; ++d)
            {
              min_ind[d]=std::min(min_ind[d], -max_ind[d]);
              max_ind[d]=std::max(-min_ind[d], max_ind[d]);
            }
          density_sptr->grow(IndexRange<3>(min_ind,max_ind));
        }
  }


  // multiplicative term
  shared_ptr<BinNormalisation> bin_norm_sptr(new TrivialBinNormalisation());
  {
    shared_ptr<ProjData> 
      mult_proj_data_sptr(new ProjDataInMemory (proj_data_sptr->get_exam_info_sptr(),
						proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone()));
    for (int seg_num=proj_data_sptr->get_min_segment_num(); 
         seg_num<=proj_data_sptr->get_max_segment_num();
         ++seg_num)
      {
        SegmentByView<float> segment = proj_data_sptr->get_empty_segment_by_view(seg_num);
        // fill in some crazy values
        float value =0;
        for (SegmentByView<float>::full_iterator iter = segment.begin_all();
             iter != segment.end_all();
             ++iter)
          {
            value = float(fabs(seg_num*value - .2)); // needs to be positive for Poisson
            *iter = value;
          }
        mult_proj_data_sptr->set_segment(segment);
      }
    bin_norm_sptr.reset(new BinNormalisationFromProjData(mult_proj_data_sptr));
  }

  // additive term
  shared_ptr<ProjData> add_proj_data_sptr(new ProjDataInMemory (proj_data_sptr->get_exam_info_sptr(),
								
proj_data_sptr->get_proj_data_info_ptr()->create_shared_clone()));
  {
    for (int seg_num=proj_data_sptr->get_min_segment_num(); 
         seg_num<=proj_data_sptr->get_max_segment_num();
         ++seg_num)
      {
        SegmentByView<float> segment = proj_data_sptr->get_empty_segment_by_view(seg_num);
        // fill in some crazy values
        float value =0;
        for (SegmentByView<float>::full_iterator iter = segment.begin_all();
             iter != segment.end_all();
             ++iter)
          {
            value = float(fabs(seg_num*value - .3)); // needs to be positive for Poisson
            *iter = value;
          }
        add_proj_data_sptr->set_segment(segment);
      }
  }

  objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>);
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>& objective_function =
    reinterpret_cast<  PoissonLogLikelihoodWithLinearModelForMeanAndProjData<target_type>& >(*objective_function_sptr);
  objective_function.set_proj_data_sptr(proj_data_sptr);
  objective_function.set_use_subset_sensitivities(true);
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
  shared_ptr<ProjectorByBinPair> proj_pair_sptr(new ProjectorByBinPairUsingProjMatrixByBin(proj_matrix_sptr));
  objective_function.set_projector_pair_sptr(proj_pair_sptr) ;
  /*
  void set_frame_num(const int);
  void set_frame_definitions(const TimeFrameDefinitions&);
  */
  objective_function.set_normalisation_sptr(bin_norm_sptr);
  objective_function.set_additive_proj_data_sptr(add_proj_data_sptr);
  objective_function.set_num_subsets(proj_data_sptr->get_num_views()/2);
  if (!check(objective_function.set_up(density_sptr)==Succeeded::yes, "set-up of objective function"))
    return;
}

void
PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests::
run_tests()
{
  cerr << "Tests for PoissonLogLikelihoodWithLinearModelForMeanAndProjData\n";

#if 1
  shared_ptr<target_type> density_sptr;
  construct_input_data(density_sptr);
  this->run_tests_for_objective_function(*this->objective_function_sptr, *density_sptr);
#else
  // alternative that gets the objective function from an OSMAPOSL .par file
  // currently disabled
  OSMAPOSLReconstruction<target_type> recon(proj_data_filename); // actually .par
  shared_ptr<GeneralisedObjectiveFunction<target_type> > objective_function_sptr = recon.get_objective_function_sptr();
  if (!check(objective_function_sptr->set_up(recon.get_initial_data_ptr())==Succeeded::yes, "set-up of objective function"))
    return;
  this->run_tests_for_objective_function(*objective_function_sptr,
                                         *recon.get_initial_data_ptr());
#endif
} 

END_NAMESPACE_STIR


USING_NAMESPACE_STIR


int main(int argc, char **argv)
{
  PoissonLogLikelihoodWithLinearModelForMeanAndProjDataTests tests(argc>1? argv[1] : 0,
                                                                   argc>2? argv[2] : 0);
  tests.run_tests();
  return tests.main_return_value();
}
