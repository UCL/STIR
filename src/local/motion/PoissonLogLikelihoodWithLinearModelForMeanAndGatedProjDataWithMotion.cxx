//
//
/*
    Copyright (C) 2005- 2012, Hammersmith Imanet Ltd
    For internal GE use only.
*/
/*!
  \file
  \ingroup recon_buildblock
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew

  \author Kris Thielemans

*/


#include "stir/recon_buildblock/DataSymmetriesForBins.h"
#include "local/stir/motion/PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"

#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
#include "local/stir/motion/Transform3DObjectImageProcessor.h"
#include "stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"

#include "stir/DiscretisedDensity.h"
// for set_projectors_and_symmetries
#include "stir/recon_buildblock/distributable.h"
// for get_symmetries_ptr()
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
// include the following to set defaults
#ifndef USE_PMRT
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#else
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
#include "stir/IO/OutputFileFormat.h"

#include <string> 
#include <iostream>
#include <utility>

START_NAMESPACE_STIR

template<typename TargetT>
const char * const 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
registered_name = 
"PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew";

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_defaults()
{
  base_type::set_defaults();

  this->_input_filename = "";
  this->max_segment_num_to_process=-1;
  // KT 20/06/2001 disabled
  //num_views_to_add=1;  
  this->_gated_proj_data_sptr.reset();
  this->zero_seg0_end_planes = 0;

  this->_additive_projection_data_filename = "0";
  this->_gated_additive_proj_data_sptr.reset();


  // set default for projector_pair_ptr
#ifndef USE_PMRT
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr
    (new ForwardProjectorByBinUsingRayTracing());
  shared_ptr<BackProjectorByBin> back_projector_ptr
    (new BackProjectorByBinUsingInterpolation());
#else
  shared_ptr<ProjMatrixByBin> PM
    (new  ProjMatrixByBinUsingRayTracing());
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr
    (new ForwardProjectorByBinUsingProjMatrixByBin(PM)); 
  shared_ptr<BackProjectorByBin> back_projector_ptr
    (new BackProjectorByBinUsingProjMatrixByBin(PM)); 
#endif

  this->projector_pair_ptr
    .reset(new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));

  // TODO at present we used a fixed size vector
  this->_normalisation_sptrs.resize(1,20);
  for (int gate_num=1; gate_num<=20; ++gate_num)
    this->_normalisation_sptrs[gate_num].reset(new TrivialBinNormalisation);
  this->frame_num = 1;
  this->frame_definition_filename = "";

  this->_forward_transformations.resize(1,20);

  // image stuff
  this->output_image_size_xy=-1;
  this->output_image_size_z=-1;
  this->zoom=1.F;
  this->Xoffset=0.F;
  this->Yoffset=0.F;
  // KT 20/06/2001 new
  this->Zoffset=0.F;

}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew Parameters");
  this->parser.add_stop_key("End PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew Parameters");
  this->parser.add_key("input filename",&this->_input_filename);

  this->parser.add_key("maximum absolute segment number to process", &this->max_segment_num_to_process);
  this->parser.add_key("zero end planes of segment 0", &this->zero_seg0_end_planes);

  // image stuff

  this->parser.add_key("zoom", &this->zoom);
  this->parser.add_key("XY output image size (in pixels)",&this->output_image_size_xy);
  this->parser.add_key("Z output image size (in pixels)",&this->output_image_size_z);
  //parser.add_key("X offset (in mm)", &this->Xoffset); // KT 10122001 added spaces
  //parser.add_key("Y offset (in mm)", &this->Yoffset);
  
  this->parser.add_key("Z offset (in mm)", &this->Zoffset);

  this->parser.add_parsing_key("Projector pair type", &this->projector_pair_ptr);
  // TODO
  this->parser.add_key("additive sinograms",&this->_additive_projection_data_filename);
  // normalisation (and attenuation correction)
  this->parser.add_key("time frame definition filename", &this->frame_definition_filename); 
  this->parser.add_key("time frame number", &this->frame_num);
  this->parser.add_parsing_key("Bin Normalisation type for gate 1", &this->_normalisation_sptrs[1]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 2", &this->_normalisation_sptrs[2]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 3", &this->_normalisation_sptrs[3]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 4", &this->_normalisation_sptrs[4]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 5", &this->_normalisation_sptrs[5]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 6", &this->_normalisation_sptrs[6]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 7", &this->_normalisation_sptrs[7]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 8", &this->_normalisation_sptrs[8]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 9", &this->_normalisation_sptrs[9]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 10", &this->_normalisation_sptrs[10]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 11", &this->_normalisation_sptrs[11]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 12", &this->_normalisation_sptrs[12]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 13", &this->_normalisation_sptrs[13]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 14", &this->_normalisation_sptrs[14]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 15", &this->_normalisation_sptrs[15]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 16", &this->_normalisation_sptrs[16]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 17", &this->_normalisation_sptrs[17]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 18", &this->_normalisation_sptrs[18]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 19", &this->_normalisation_sptrs[19]);
  this->parser.add_parsing_key("Bin Normalisation type for gate 20", &this->_normalisation_sptrs[20]);



  this->parser.add_parsing_key("transformation type for gate 1", 
                               &this->_forward_transformations[1]);
  this->parser.add_parsing_key("transformation type for gate 2", 
                               &this->_forward_transformations[2]);
  this->parser.add_parsing_key("transformation type for gate 3", 
                               &this->_forward_transformations[3]);
  this->parser.add_parsing_key("transformation type for gate 4", 
                               &this->_forward_transformations[4]);
  this->parser.add_parsing_key("transformation type for gate 5", 
                               &this->_forward_transformations[5]);
  this->parser.add_parsing_key("transformation type for gate 6", 
                               &this->_forward_transformations[6]);
  this->parser.add_parsing_key("transformation type for gate 7", 
                               &this->_forward_transformations[7]);
  this->parser.add_parsing_key("transformation type for gate 8", 
                               &this->_forward_transformations[8]);
  this->parser.add_parsing_key("transformation type for gate 9", 
                               &this->_forward_transformations[9]);
  this->parser.add_parsing_key("transformation type for gate 10", 
                               &this->_forward_transformations[10]);
  this->parser.add_parsing_key("transformation type for gate 11", 
                               &this->_forward_transformations[11]);
  this->parser.add_parsing_key("transformation type for gate 12", 
                               &this->_forward_transformations[12]);
  this->parser.add_parsing_key("transformation type for gate 13", 
                               &this->_forward_transformations[13]);
  this->parser.add_parsing_key("transformation type for gate 14", 
                               &this->_forward_transformations[14]);
  this->parser.add_parsing_key("transformation type for gate 15", 
                               &this->_forward_transformations[15]);
  this->parser.add_parsing_key("transformation type for gate 16", 
                               &this->_forward_transformations[16]);
  this->parser.add_parsing_key("transformation type for gate 17", 
                               &this->_forward_transformations[17]);
  this->parser.add_parsing_key("transformation type for gate 18", 
                               &this->_forward_transformations[18]);
  this->parser.add_parsing_key("transformation type for gate 19", 
                               &this->_forward_transformations[19]);
  this->parser.add_parsing_key("transformation type for gate 20", 
                               &this->_forward_transformations[20]);


}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  if (this->_input_filename.length() == 0)
  { warning("You need to specify an input file"); return true; }
 
  this->_gated_proj_data_sptr.reset(GatedProjData::read_from_file(this->_input_filename));

 // image stuff
  if (this->zoom <= 0)
  { warning("zoom should be positive"); return true; }
  
  if (this->output_image_size_xy!=-1 && this->output_image_size_xy<1) // KT 10122001 appended_xy
  { warning("output image size xy must be positive (or -1 as default)"); return true; }
  if (this->output_image_size_z!=-1 && this->output_image_size_z<1) // KT 10122001 new
  { warning("output image size z must be positive (or -1 as default)"); return true; }


  if (this->_additive_projection_data_filename != "0")
  {
    this->_gated_additive_proj_data_sptr
      .reset(GatedProjData::read_from_file(this->_additive_projection_data_filename));
  };

  // read time frame def 
   if (this->frame_definition_filename.size()!=0)
    this->frame_defs = TimeFrameDefinitions(this->frame_definition_filename);
   else
    {
      // make a single frame starting from 0 to 1.
      std::vector<std::pair<double, double> > frame_times(1, std::pair<double,double>(0,1));
      this->frame_defs = TimeFrameDefinitions(frame_times);
    } 

  return false;
}

template <typename TargetT>
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew()
{
  this->set_defaults();
}

template <typename TargetT>
TargetT *
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
construct_target_ptr() const
{
  return
      new VoxelsOnCartesianGrid<float> (*this->_gated_proj_data_sptr->get_proj_data_info_ptr(),
                                        static_cast<float>(this->zoom),
                                        CartesianCoordinate3D<float>(static_cast<float>(this->Zoffset),
                                                                     static_cast<float>(this->Yoffset),
                                                                     static_cast<float>(this->Xoffset)),
                                        CartesianCoordinate3D<int>(this->output_image_size_z,
                                                                   this->output_image_size_xy,
                                                                   this->output_image_size_xy)
                                       );
}

/***************************************************************
  set_ functions
***************************************************************/

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_proj_data_sptr(const shared_ptr<GatedProjData>& arg)
{
  this->_gated_proj_data_sptr = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_max_segment_num_to_process(const int arg)
{
  this->max_segment_num_to_process = arg;

}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_zero_seg0_end_planes(const bool arg)
{
  this->zero_seg0_end_planes = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_additive_proj_data_sptr(const shared_ptr<GatedProjData>& arg)
{

  this->_gated_additive_proj_data_sptr = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_projector_pair_sptr(const shared_ptr<ProjectorByBinPair>& arg) 
{
  this->projector_pair_ptr = arg;

}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_frame_num(const int arg)
{
  this->frame_num = arg;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_frame_definitions(const TimeFrameDefinitions& arg)
{
  this->frame_defs = arg;
}

/***************************************************************
  set_up()
***************************************************************/
template<typename TargetT>
Succeeded 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
set_up_before_sensitivity(shared_ptr<TargetT > const& target_sptr)
{
  shared_ptr<ProjDataInfo> 
    proj_data_info_sptr(this->_gated_proj_data_sptr->get_proj_data_info_ptr()->clone());

  if (this->max_segment_num_to_process==-1)
    this->max_segment_num_to_process =
      proj_data_info_sptr->get_max_segment_num();

  if (this->max_segment_num_to_process > proj_data_info_sptr->get_max_segment_num()) 
    { 
      warning("max_segment_num_to_process (%d) is too large",
              this->max_segment_num_to_process); 
      return Succeeded::no;
    }

  proj_data_info_sptr->
    reduce_segment_range(-this->max_segment_num_to_process,
                         +this->max_segment_num_to_process);
  
  if (is_null_ptr(this->projector_pair_ptr))
    { warning("You need to specify a projector pair"); return Succeeded::no; }

  // set projectors to be used for the calculations

  this->projector_pair_ptr->set_up(proj_data_info_sptr, 
                                   target_sptr);

  // TODO check compatibility between symmetries for forward and backprojector
  this->symmetries_sptr.reset(
	  this->projector_pair_ptr->get_back_projector_sptr()->get_symmetries_used()->clone());

  // initialise the objective functions for each gate
  {
    // for sensitivity work-around  (see below)
    shared_ptr<TargetT> empty_target_sptr(target_sptr->get_empty_copy());

    this->_functions.resize(this->_gated_proj_data_sptr->get_num_gates());
    for (unsigned int gate_num=1; 
         gate_num<=this->_gated_proj_data_sptr->get_num_gates(); 
         ++gate_num)
      {
      
        PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>
          objective_function;

        objective_function.
          set_proj_data_sptr(this->_gated_proj_data_sptr->get_proj_data_sptr(gate_num));
        objective_function.
          set_max_segment_num_to_process(this->max_segment_num_to_process);
        objective_function.
          set_zero_seg0_end_planes(this->zero_seg0_end_planes);

        {
          shared_ptr<ProjectorByBinPair> projector_pair_sptr_this_gate;

          if (is_null_ptr(this->_forward_transformations[gate_num]))
            {
              projector_pair_sptr_this_gate =
                this->projector_pair_ptr;
            }
          else
            {
              Transform3DObjectImageProcessor<float> const * forward_transformer_ptr =
                dynamic_cast<Transform3DObjectImageProcessor<float> const *>
                (this->_forward_transformations[gate_num].get());
              if (forward_transformer_ptr==0)
                {
                  warning("transformation type has to be Transform3DObjectImageProcessor");
                  return Succeeded::no;
                }
              shared_ptr<ForwardProjectorByBin> forward_projector_sptr_this_gate
		(
		 new PresmoothingForwardProjectorByBin(this->projector_pair_ptr->
						       get_forward_projector_sptr(),
						       this->_forward_transformations[gate_num])
		 );

              shared_ptr<DataProcessor<TargetT> > transpose_transformer_sptr
		(
                //forward_transformer_ptr->clone()
		 new Transform3DObjectImageProcessor<float>(*forward_transformer_ptr)
		 );
              // TODO get rid if dynamic cast when using boost::shared_ptr
              Transform3DObjectImageProcessor<float> & transpose_transformer =
                dynamic_cast<Transform3DObjectImageProcessor<float> &>(*transpose_transformer_sptr);
              transpose_transformer.
                set_do_transpose(!forward_transformer_ptr->get_do_transpose());
              shared_ptr<BackProjectorByBin> back_projector_sptr_this_gate
		(new PostsmoothingBackProjectorByBin(this->projector_pair_ptr->
						     get_back_projector_sptr(),
						     transpose_transformer_sptr)
		 );
              projector_pair_sptr_this_gate.
		reset(new ProjectorByBinPairUsingSeparateProjectors
		      (forward_projector_sptr_this_gate,
		       back_projector_sptr_this_gate));
            }
          objective_function.
            set_projector_pair_sptr(projector_pair_sptr_this_gate);
        }
        if (is_null_ptr(this->_gated_additive_proj_data_sptr))
           {
	     shared_ptr<ProjData> nullsptr;
             objective_function. 
               set_additive_proj_data_sptr(nullsptr);
           }
        else
          {
            objective_function.
              set_additive_proj_data_sptr(this->_gated_additive_proj_data_sptr->get_proj_data_sptr(gate_num));
          }
        objective_function.
          set_frame_num(this->frame_num);
        objective_function.
          set_frame_definitions(this->frame_defs);
        objective_function.
          set_normalisation_sptr(this->_normalisation_sptrs[gate_num]);
        objective_function.
          set_num_subsets(this->num_subsets);

#if 0
        // we need to prevent computation of the subsensitivities at present
        // we set them to an empty image.
        // WARNING this will create serious problems if we call the actual gradient
        // of this objective_function
          {
            objective_function.
              set_recompute_sensitivity(false);
            objective_function.set_use_subset_sensitivities(false);
            objective_function.
              set_sensitivity_sptr(empty_target_sptr, 0);
          }
#endif
          if (objective_function.set_up(empty_target_sptr) != Succeeded::yes)
            error("Single gate objective functions is not set-up correctly!");
        
          // TODO dangerous for -1
          this->_functions[gate_num-1] = objective_function;

      }
  }
#if 0
  if (base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;
#endif

  return Succeeded::yes;
}

/***************************************************************
  functions that compute the value/gradient of the objective function etc
***************************************************************/

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
                                                      const TargetT &current_estimate, 
                                                      const int subset_num)
{
  typename base_type::_functions_iterator_type iter = this->_functions.begin();
  typename base_type::_functions_iterator_type end_iter = this->_functions.end();
  if (iter != end_iter)
    {
      iter->compute_sub_gradient_without_penalty_plus_sensitivity(gradient, 
                                                                  current_estimate, 
                                                                  subset_num);
    }
  ++iter;
  if (iter == end_iter)
    return;

  // TODO!!!!!!!!
  shared_ptr<TargetT> gradient_this_function_sptr(gradient.get_empty_copy());

  while (iter != end_iter)
    {
      iter->compute_sub_gradient_without_penalty_plus_sensitivity(*gradient_this_function_sptr, 
                                                                  current_estimate, 
                                                                  subset_num);
      // now add it to the total gradient
      typename TargetT::full_iterator gradient_iter = 
        gradient.begin_all();
      typename TargetT::full_iterator gradient_end = 
        gradient.end_all();
      typename TargetT::const_full_iterator gradient_this_function_iter =
        gradient_this_function_sptr ->begin_all_const();
      while (gradient_iter != gradient_end)
        {
          *gradient_iter++ += *gradient_this_function_iter++;
        }
      ++iter;
    }
  

}


template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
  typename base_type::_functions_const_iterator_type iter = this->_functions.begin();
  typename base_type::_functions_const_iterator_type end_iter = this->_functions.end();
  while (iter != end_iter)
    {
      iter->add_subset_sensitivity(sensitivity,
                                   subset_num);

      ++iter;
    }

}


#  ifdef _MSC_VER
// prevent warning message on instantiation of abstract class 
#  pragma warning(disable:4661)
#  endif

template class PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotionNew<DiscretisedDensity<3,float> >;

END_NAMESPACE_STIR

