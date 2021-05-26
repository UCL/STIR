//
//
/*
    Copyright (C) 2006- 2007, Hammersmith Imanet Ltd
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
  \ingroup GeneralisedObjectiveFunction
  \brief Implementation of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData

  \author Charalampos Tsoumpas

*/
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/stream.h"

#include "stir/recon_buildblock/ProjectorByBinPair.h"

// for get_symmetries_ptr()
#include "stir/DataSymmetriesForViewSegmentNumbers.h"
// include the following to set defaults
#ifndef USE_PMRT
#  include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#  include "stir/recon_buildblock/BackProjectorByBinUsingInterpolation.h"
#else
#  include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#  include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#  include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"

#include "stir/Succeeded.h"
//#include "stir/IO/OutputFileFormat.h"
//#include <algorithm>
#include <string>

#include "stir_experimental/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData.h"

START_NAMESPACE_STIR

template <typename TargetT>
const char* const PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::registered_name =
    "PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData";

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::set_defaults() {
  base_type::set_defaults();

  this->_input_filename = "";
  this->_max_segment_num_to_process = 0;
  // num_views_to_add=1;    // KT 20/06/2001 disabled
  this->_dyn_proj_data_sptr = NULL;
  this->_zero_seg0_end_planes = 0;
  this->_additive_dyn_proj_data_filename = "0";
  this->_additive_dyn_proj_data_sptr = NULL;

#ifndef USE_PMRT // set default for _projector_pair_ptr
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr = new ForwardProjectorByBinUsingRayTracing();
  shared_ptr<BackProjectorByBin> back_projector_ptr = new BackProjectorByBinUsingInterpolation();
#else
  shared_ptr<ProjMatrixByBin> PM = new ProjMatrixByBinUsingRayTracing();
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr = new ForwardProjectorByBinUsingProjMatrixByBin(PM);
  shared_ptr<BackProjectorByBin> back_projector_ptr = new BackProjectorByBinUsingProjMatrixByBin(PM);
#endif

  this->_projector_pair_ptr = new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr);
  this->_normalisation_sptr = new TrivialBinNormalisation;

  // image stuff
  this->_output_image_size_xy = -1;
  this->_output_image_size_z = -1;
  this->_zoom = 1.F;
  this->_Xoffset = 0.F;
  this->_Yoffset = 0.F;
  this->_Zoffset = 0.F; // KT 20/06/2001 new
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::initialise_keymap() {
  base_type::initialise_keymap();
  this->parser.add_start_key("PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData Parameters");
  this->parser.add_stop_key("End PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData Parameters");
  this->parser.add_key("input file", &this->_input_filename);

  // parser.add_key("mash x views", &num_views_to_add);   // KT 20/06/2001 disabled
  this->parser.add_key("maximum absolute segment number to process", &this->_max_segment_num_to_process);
  this->parser.add_key("zero end planes of segment 0", &this->_zero_seg0_end_planes);

  // image stuff
  this->parser.add_key("zoom", &this->_zoom);
  this->parser.add_key("XY output image size (in pixels)", &this->_output_image_size_xy);
  this->parser.add_key("Z output image size (in pixels)", &this->_output_image_size_z);

  // parser.add_key("X offset (in mm)", &this->Xoffset); // KT 10122001 added spaces
  // parser.add_key("Y offset (in mm)", &this->Yoffset);
  this->parser.add_key("Z offset (in mm)", &this->_Zoffset);
  this->parser.add_parsing_key("Projector pair type", &this->_projector_pair_ptr);

  // Scatter correction
  this->parser.add_key("additive sinograms", &this->_additive_dyn_proj_data_filename);

  // normalisation (and attenuation correction)
  this->parser.add_parsing_key("Bin Normalisation type", &this->_normalisation_sptr);
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::post_processing() {
  if (base_type::post_processing() == true)
    return true;
  if (this->_input_filename.length() == 0) {
    warning("You need to specify an input file");
    return true;
  }

#if 0 // KT 20/06/2001 disabled as not functional yet
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
  { warning("The 'mash x views' key has an invalid value (must be 1 or even number)"); return true; }
#endif
  this->_dyn_proj_data_sptr = DynamicProjData::read_from_file(_input_filename);

  // image stuff
  if (this->_zoom <= 0) {
    warning("zoom should be positive");
    return true;
  }

  if (this->_output_image_size_xy != -1 && this->_output_image_size_xy < 1) // KT 10122001 appended_xy
  {
    warning("output image size xy must be positive (or -1 as default)");
    return true;
  }
  if (this->_output_image_size_z != -1 && this->_output_image_size_z < 1) // KT 10122001 new
  {
    warning("output image size z must be positive (or -1 as default)");
    return true;
  }

  if (this->_additive_dyn_proj_data_filename != "0") {
    cerr << "\nReading additive projdata data " << this->_additive_dyn_proj_data_filename << endl;
    this->_additive_dyn_proj_data_sptr = DynamicProjData::read_from_file(this->_additive_dyn_proj_data_filename);
  }
  return false;
}
#if 0 // ChT::ToDo
template <typename TargetT>
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData()
{
  this->set_defaults();
}

template <typename TargetT>
TargetT *
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
construct_target_ptr() const
{
#  if 0 
  const shared_ptr<DiscretisedDensity<3,float> > density_template_sptr = 
    (*(this->_dyn_proj_data_sptr->get_proj_data_info_sptr()),
     static_cast<float>(this->_zoom),
     CartesianCoordinate3D<float>(static_cast<float>(this->_Zoffset),
				  static_cast<float>(this->_Yoffset),
				  static_cast<float>(this->_Xoffset)),
     CartesianCoordinate3D<int>(this->_output_image_size_z,
				this->_output_image_size_xy,
				this->_output_image_size_xy));
  
  const shared_ptr<Scanner> scanner_sptr = new Scanner(*(this->_proj_data_info_sptr->get_scanner_ptr()));
  
  return
    new DynamicDiscretisedDensity(this->_frame_defs(),scanner_sptr,density_template_sptr);
#  else
  return 0;
#  endif 
}
/***************************************************************
  subset balancing
 ***************************************************************/

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{  // call actual_subsets_are_approximately_balanced( for first single_frame_obj_func
  if (this->_frame_defs.get_num_frames() == 0 || this->_single_frame_obj_funcs.size() == 0)
    error("PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData:\n"
	  "actual_subsets_are_approximately_balanced called but not frames yet.\n");
  else if(this->_single_frame_obj_funcs.size() != 0)
    {
      bool frames_are_balanced=true;
      for(unsigned int frame_num=1 ; frame_num<=this->_frame_defs.get_num_frames() ; ++frame_num)
	frames_are_balanced &= this->_single_frame_obj_funcs[frame_num].subsets_are_approximately_balanced(warning_message);
      return frames_are_balanced;
    }
  else 
      warning("Something stange happened in PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData:\n"
	      "actual_subsets_are_approximately_balanced \n");
    return 
      false;    
}

template<typename TargetT>
int
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
set_num_subsets(const int num_subsets)
{
  for(unsigned int frame_num=1;frame_num<=this->_frame_defs.get_num_frames();++frame_num)
    {
      if(this->_single_frame_obj_funcs.size() != 0)
       if(this->_single_frame_obj_funcs[frame_num].set_num_subsets(num_subsets) != num_subsets)
	 error("set_num_subsets didn't work");
    }
  this->num_subsets=num_subsets;
  return this->num_subsets;
}

/***************************************************************
  set_up()
***************************************************************/
template<typename TargetT>
Succeeded 
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
set_up(shared_ptr<TargetT > const& target_sptr)
{
  if (base_type::set_up(target_sptr) != Succeeded::yes)
    return Succeeded::no;

  if (this->_max_segment_num_to_process==-1)
    this->_max_segment_num_to_process =
      (this->_dyn_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num();

  if (this->_max_segment_num_to_process > (this->_dyn_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num()) 
    { 
      warning("_max_segment_num_to_process (%d) is too large",
	      this->_max_segment_num_to_process); 
      return Succeeded::no;
    }

  shared_ptr<ProjDataInfo> proj_data_info_sptr =
    (this->_dyn_proj_data_sptr->get_proj_data_sptr(1))->get_proj_data_info_sptr()->clone();
  proj_data_info_sptr->
    reduce_segment_range(-this->_max_segment_num_to_process,
			 +this->_max_segment_num_to_process);
  
  if (is_null_ptr(this->_projector_pair_ptr))
    { warning("You need to specify a projector pair"); return Succeeded::no; }

  if (this->num_subsets <= 0)
    {
      warning("Number of subsets %d should be larger than 0.",
	      this->num_subsets);
      return Succeeded::no;
    }

  if (is_null_ptr(this->_normalisation_sptr))
  {
    warning("Invalid normalisation object");
    return Succeeded::no;
  }

  if (this->_normalisation_sptr->set_up(proj_data_info_sptr) == Succeeded::no)
    return Succeeded::no;
  const shared_ptr<DiscretisedDensity<3,float> > density_template_sptr = (target_sptr->get_density(1)).clone();

   {
     // construct _single_frame_obj_funcs
     this->_single_frame_obj_funcs.resize(1,target_sptr->get_time_frame_definitions().get_num_frames());   
     for(unsigned int frame_num=1;frame_num<=target_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
       {
	 this->_single_frame_obj_funcs[frame_num].set_projector_pair_sptr(this->_projector_pair_ptr);
	 this->_single_frame_obj_funcs[frame_num].set_proj_data_sptr(this->_dyn_proj_data_sptr->get_proj_data_sptr(frame_num));
	 this->_single_frame_obj_funcs[frame_num].set_max_segment_num_to_process(this->_max_segment_num_to_process);
	 this->_single_frame_obj_funcs[frame_num].set_zero_seg0_end_planes(this->_zero_seg0_end_planes!=0);
	 if(this->_additive_dyn_proj_data_sptr!=NULL)
	   this->_single_frame_obj_funcs[frame_num].set_additive_proj_data_sptr(this->_additive_dyn_proj_data_sptr->get_proj_data_sptr(frame_num));
	 this->_single_frame_obj_funcs[frame_num].set_num_subsets(this->num_subsets);
	 this->_single_frame_obj_funcs[frame_num].set_frame_num(frame_num);
	 this->_single_frame_obj_funcs[frame_num].set_frame_definitions(target_sptr->get_time_frame_definitions());
	 this->_single_frame_obj_funcs[frame_num].set_normalisation_sptr(this->_normalisation_sptr);
	 this->_single_frame_obj_funcs[frame_num].set_recompute_sensitivity(this->recompute_sensitivity);
	 if(this->_single_frame_obj_funcs[frame_num].set_up(density_template_sptr) != Succeeded::yes)
	   error("Single frame objective functions is not set correctly!");
       }
  /* TODO
     current implementation of compute_sub_gradient does not use subsensitivity
  */ 
   }//_single_frame_obj_funcs[frame_num]
   if(!this->subsets_are_approximately_balanced())
     {
       warning("Number of subsets %d is such that subsets will be very unbalanced.\n"
	       "Current implementation of PoissonLogLikelihoodWithLinearModelForMean cannot handle this.",
	       this->num_subsets);
       return Succeeded::no;
     }

  if(this->recompute_sensitivity)
    {
      std::cerr << "Computing sensitivity" << std::endl;
      for(unsigned int frame_num=1; frame_num<=this->_frame_defs.get_num_frames(); ++frame_num)
	{
	  (target_sptr->get_density(frame_num)).compute_sensitivities();
	}

      std::cerr << "Done computing sensitivity" << std::endl;
      if (this->sensitivity_filename.size()!=0)
	{  // TODO writes only first if use_subset_sensitivities
	    target_sptr->get_density[frame_num]=this->get_sensitivity(0).;
	    ((target_sptr->get_density(frame_num)).get_sensitivity(0)).write_to_ecat7(this->sensitivity_filename);

	}
    }
  return Succeeded::yes;
}

/*************************************************************************
  functions that compute the value/gradient of the objective function etc
*************************************************************************/

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
						      const TargetT &current_estimate, 
						      const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);
  for(unsigned int frame_num=1; frame_num<=this->_frame_defs.get_num_frames(); ++frame_num)
    this->_single_frame_obj_funcs[frame_num].
      compute_sub_gradient_without_penalty_plus_sensitivity(gradient[frame_num], 
							    current_estimate[frame_num], 
							    subset_num);
}

template<typename TargetT>
float
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
						  const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  float result = 0;
  // loop over single_frame and use model_matrix
  for(unsigned int frame_num=1; frame_num<=this->_frame_defs.get_num_frames(); ++frame_num)
    result +=
      this->_single_frame_obj_funcs[frame_num].
      compute_objective_function_without_penalty(current_estimate[frame_num], 
						 subset_num);   
  return result;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
  // TODO this will NOT add to the subset sensitivity, but overwrite
  // loop over single_frame 
  for(unsigned int frame_num=1 ; frame_num<=this->_frame_defs.get_num_frames() ; ++frame_num)
    {    
      sensitivity[frame_num]=this->_single_frame_obj_funcs[frame_num].get_sensitivity(subset_num);
      //  add_subset_sensitivity(dyn_sensitivity[frame_num],subset_num);
    }
}
#endif // ChT::ToDo

#ifdef _MSC_VER
// prevent warning message on instantiation of abstract class
#  pragma warning(disable : 4661)
#endif

template class PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<DynamicDiscretisedDensity>;

END_NAMESPACE_STIR
