/*
 Copyright (C) 2006- 2009, Hammersmith Imanet Ltd
 Copyright (C) 2011 - 2013, King's College London
 This file is part of STIR.
 
 This file is free software; you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation; either version 2.3 of the License, or
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
  \brief Implementation of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion

  \author Charalampos Tsoumpas
  \author Kris Thielemans
  \author Sanida Mustafovic

*/
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/info.h"

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

#include <algorithm>
#include <string> 
// For Motion
#include "stir/spatial_transformation/GatedSpatialTransformation.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"

START_NAMESPACE_STIR

template<typename TargetT>
const char * const 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
registered_name = 
"PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion";

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
set_defaults()
{
  base_type::set_defaults();

  this->_input_filename="";
  this->_max_segment_num_to_process=-1; // use all segments
  //num_views_to_add=1;    // KT 20/06/2001 disabled

  this->_gated_proj_data_sptr.reset(); 
  this->_zero_seg0_end_planes = 0;
  this->_reverse_motion_vectors_filename_prefix="0";
  this->_normalisation_gated_proj_data_filename="1";
  this->_normalisation_gated_proj_data_sptr.reset();
  //  this->_reverse_motion_vectors_sptr=NULL;
  this->_motion_vectors_filename_prefix="0";
  //  this->_motion_vectors_sptr=NULL;
  this->_gate_definitions_filename="0";
  // this->_time_gate_definitions_sptr=NULL;
  this->_additive_gated_proj_data_filename = "0";
  this->_additive_gated_proj_data_sptr.reset();

#ifndef USE_PMRT // set default for _projector_pair_ptr
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

  this->_projector_pair_ptr.reset(
                                  new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));

  // image stuff
  this->_output_image_size_xy=-1;
  this->_output_image_size_z=-1;
  this->_zoom=1.F;
  this->_Xoffset=0.F;
  this->_Yoffset=0.F;
  this->_Zoffset=0.F;   // KT 20/06/2001 new
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion Parameters");
  this->parser.add_stop_key("End PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion Parameters");
  this->parser.add_key("input filename",&this->_input_filename);

  // parser.add_key("mash x views", &num_views_to_add);   // KT 20/06/2001 disabled
  this->parser.add_key("maximum absolute segment number to process", &this->_max_segment_num_to_process);
  this->parser.add_key("zero end planes of segment 0", &this->_zero_seg0_end_planes);

  // image stuff
  this->parser.add_key("zoom", &this->_zoom);
  this->parser.add_key("XY output image size (in pixels)",&this->_output_image_size_xy);
  this->parser.add_key("Z output image size (in pixels)",&this->_output_image_size_z);

  // parser.add_key("X offset (in mm)", &this->Xoffset); // KT 10122001 added spaces
  // parser.add_key("Y offset (in mm)", &this->Yoffset);
  this->parser.add_key("Z offset (in mm)", &this->_Zoffset);
  this->parser.add_parsing_key("Projector pair type", &this->_projector_pair_ptr);

  // Scatter correction
  this->parser.add_key("additive sinograms",&this->_additive_gated_proj_data_filename);

  // normalisation (and attenuation correction)
  this->parser.add_key("normalisation sinograms", &this->_normalisation_gated_proj_data_filename);
  
  // Motion Information
  this->parser.add_key("Gate Definitions filename", &this->_gate_definitions_filename);
  this->parser.add_key("Motion Vectors filename prefix", &this->_motion_vectors_filename_prefix);
  this->parser.add_key("Reverse Motion Vectors filename prefix", &this->_reverse_motion_vectors_filename_prefix); 
}

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  if (this->_input_filename.length() == 0)
    { warning("You need to specify an input filename"); return true; }
  
  this->_gated_proj_data_sptr.reset(GatedProjData::read_from_file(this->_input_filename));
  
  // image stuff
  if (this->_zoom <= 0)
    { warning("zoom should be positive"); return true; }
  
  if (this->_output_image_size_xy!=-1 && this->_output_image_size_xy<1) // KT 10122001 appended_xy
    { warning("output image size xy must be positive (or -1 as default)"); return true; }
  if (this->_output_image_size_z!=-1 && this->_output_image_size_z<1) // KT 10122001 new
    { warning("output image size z must be positive (or -1 as default)"); return true; }

  if (this->_additive_gated_proj_data_filename != "0")
    {
      info(boost::format("Reading additive projdata data %1%") % this->_additive_gated_proj_data_filename);
      this->_additive_gated_proj_data_sptr.reset(
                                                 GatedProjData::read_from_file(this->_additive_gated_proj_data_filename));
    }
  if (this->_normalisation_gated_proj_data_filename != "1")
    {
      info(boost::format("Reading normalisation projdata data %1%") % this->_normalisation_gated_proj_data_filename);
      this->_normalisation_gated_proj_data_sptr.reset(
                                                      GatedProjData::read_from_file(this->_normalisation_gated_proj_data_filename));
    }

  this->_time_gate_definitions.read_gdef_file(this->_gate_definitions_filename);

  if (this->_reverse_motion_vectors_filename_prefix != "0")
    this->_reverse_motion_vectors.read_from_files(this->_reverse_motion_vectors_filename_prefix);
  if (this->_motion_vectors_filename_prefix != "0")
    this->_motion_vectors.read_from_files(this->_motion_vectors_filename_prefix);
  return false;
}

template <typename TargetT>
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion()
{
  this->set_defaults();
}

template <typename TargetT>
TargetT *
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
construct_target_ptr() const
{  
  return 
    new VoxelsOnCartesianGrid<float> (*this->_gated_proj_data_sptr->get_proj_data_info_ptr(),
                                      static_cast<float>(this->_zoom), 
                                      CartesianCoordinate3D<float>(static_cast<float>(this->_Zoffset), 
                                                                   static_cast<float>(this->_Yoffset), 
                                                                   static_cast<float>(this->_Xoffset)), 
                                      CartesianCoordinate3D<int>(this->_output_image_size_z, 
                                                                 this->_output_image_size_xy, 
                                                                 this->_output_image_size_xy)
                                      ); 
}
/***************************************************************
  subset balancing
***************************************************************/

template<typename TargetT>
bool
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{  // call actual_subsets_are_approximately_balanced() for first single_gate_obj_func
  if (this->get_time_gate_definitions().get_num_gates() == 0 || this->_single_gate_obj_funcs.size() == 0)
    error("PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion:\n"
	  "actual_subsets_are_approximately_balanced called but no gates yet.");
  else if(this->_single_gate_obj_funcs.size() != 0)
    {
      bool gates_are_balanced=true;
      for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
        gates_are_balanced &= this->_single_gate_obj_funcs[gate_num].subsets_are_approximately_balanced(warning_message);
      return gates_are_balanced;
    }
  else 
    error("Something strange happened in PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion:\n"
            "actual_subsets_are_approximately_balanced called before setup()?");
  return 
    false;    
}

/***************************************************************
  get_ functions
***************************************************************/
template <typename TargetT>
const TimeGateDefinitions &
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_time_gate_definitions() const
{	return this->_time_gate_definitions; }

template <typename TargetT>
const GatedProjData& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_gated_proj_data() const
{ return *this->_gated_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<GatedProjData>& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_gated_proj_data_sptr() const
{ return this->_gated_proj_data_sptr; }

template <typename TargetT>
const int 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_max_segment_num_to_process() const
{ return this->_max_segment_num_to_process; }

template <typename TargetT>
const bool 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_zero_seg0_end_planes() const
{ return this->_zero_seg0_end_planes; }

template <typename TargetT>
const GatedProjData& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_additive_gated_proj_data() const
{ return *this->_additive_gated_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<GatedProjData>& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_additive_gated_proj_data_sptr() const
{ return this->_additive_gated_proj_data_sptr; }

template <typename TargetT>
const GatedProjData& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_normalisation_gated_proj_data() const
{ return *this->_normalisation_gated_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<GatedProjData>& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_normalisation_gated_proj_data_sptr() const
{ return this->_normalisation_gated_proj_data_sptr; }

template <typename TargetT>
const ProjectorByBinPair& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_projector_pair() const
{ return *this->_projector_pair_ptr; }

template <typename TargetT>
const shared_ptr<ProjectorByBinPair>& 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_projector_pair_sptr() const
{ return this->_projector_pair_ptr; }

/***************************************************************
  set_ functions
***************************************************************/
template<typename TargetT>
int
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
set_num_subsets(const int num_subsets)
{
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
    {
      if(this->_single_gate_obj_funcs.size() != 0)
		if(this->_single_gate_obj_funcs[gate_num].set_num_subsets(num_subsets) != num_subsets)
			error("set_num_subsets didn't work");
    }
  this->num_subsets=num_subsets;
  return this->num_subsets;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
set_time_gate_definitions(const TimeGateDefinitions & time_gate_definitions)
{ this->_time_gate_definitions=time_gate_definitions; }

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
set_input_data(const shared_ptr<ExamInfo> &)
{
    error("non implemented yet");
}

/***************************************************************
  set_up()
***************************************************************/
template<typename TargetT>
Succeeded 
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
set_up_before_sensitivity(shared_ptr<TargetT > const& target_sptr)
{
  /*!todo define in the PoissonLogLikelihoodWithLinearModelForMean class to return Succeeded::yes 
    if (base_type::set_up_before_sensitivity(target_sptr) != Succeeded::yes)
    return Succeeded::no;
  */
  if (this->_max_segment_num_to_process==-1)
    this->_max_segment_num_to_process =
      (this->_gated_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num();

  if (this->_max_segment_num_to_process > (this->_gated_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num()) 
    { 
      warning("_max_segment_num_to_process (%d) is too large",
	      this->_max_segment_num_to_process); 
      return Succeeded::no;
    }

  shared_ptr<ProjDataInfo> proj_data_info_sptr(
                                               (this->_gated_proj_data_sptr->get_proj_data_sptr(1))->get_proj_data_info_ptr()->clone());
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
  {
    const shared_ptr<DiscretisedDensity<3,float> > density_template_sptr(target_sptr->get_empty_copy()); // target_sptr appears not to be set up correctly
    const shared_ptr<Scanner> scanner_sptr(new Scanner(*proj_data_info_sptr->get_scanner_ptr()));
    this->_gated_image_template=GatedDiscretisedDensity(this->get_time_gate_definitions(), density_template_sptr);

    // construct _single_gate_obj_funcs
    this->_single_gate_obj_funcs.resize(1,this->get_time_gate_definitions().get_num_gates());
	   
    for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
      {
        info(boost::format("Objective Function for Gate Number: %1%") % gate_num);
	this->_single_gate_obj_funcs[gate_num].set_projector_pair_sptr(this->_projector_pair_ptr);
	this->_single_gate_obj_funcs[gate_num].set_proj_data_sptr(this->_gated_proj_data_sptr->get_proj_data_sptr(gate_num));
	this->_single_gate_obj_funcs[gate_num].set_max_segment_num_to_process(this->_max_segment_num_to_process);
	this->_single_gate_obj_funcs[gate_num].set_zero_seg0_end_planes(this->_zero_seg0_end_planes!=0);
	if(this->_additive_gated_proj_data_sptr!=NULL)
	  this->_single_gate_obj_funcs[gate_num].set_additive_proj_data_sptr(this->_additive_gated_proj_data_sptr->get_proj_data_sptr(gate_num));
	this->_single_gate_obj_funcs[gate_num].set_num_subsets(this->num_subsets);
	this->_single_gate_obj_funcs[gate_num].set_frame_num(1);//This should be gate...
	std::vector<std::pair<double, double> > frame_times(1, std::pair<double,double>(0,1));
	this->_single_gate_obj_funcs[gate_num].set_frame_definitions(TimeFrameDefinitions(frame_times));

	shared_ptr<BinNormalisation> current_gate_norm_factors_sptr;
	if (is_null_ptr(this->_normalisation_gated_proj_data_sptr))
	  current_gate_norm_factors_sptr.reset(new TrivialBinNormalisation);
	else	{
          shared_ptr<ProjData> norm_data_sptr(this->_normalisation_gated_proj_data_sptr->get_proj_data_sptr(gate_num));
          current_gate_norm_factors_sptr.reset( 
                                               new BinNormalisationFromProjData(norm_data_sptr));
	}		
	this->_single_gate_obj_funcs[gate_num].set_normalisation_sptr(current_gate_norm_factors_sptr);
	this->_single_gate_obj_funcs[gate_num].set_recompute_sensitivity(this->get_recompute_sensitivity());
	this->_single_gate_obj_funcs[gate_num].set_use_subset_sensitivities(this->get_use_subset_sensitivities());

	if(this->_single_gate_obj_funcs[gate_num].set_up(density_template_sptr) != Succeeded::yes)
	  error("Single gate objective functions is not set correctly!");
      }
  }//_single_gate_obj_funcs[gate_num]
  return Succeeded::yes;
}

/*************************************************************************
  functions that compute the value/gradient of the objective function etc
*************************************************************************/

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
						      const TargetT &current_estimate, 
						      const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  GatedDiscretisedDensity gated_gradient=this->_gated_image_template;
  GatedDiscretisedDensity gated_image_estimate=this->_gated_image_template;
  // The following initialization doesn't stabilize reconstruction.
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
    std::fill(gated_image_estimate[gate_num].begin_all(),
	      gated_image_estimate[gate_num].end_all(),
	      0.F);		  
  this->_motion_vectors.warp_image(gated_image_estimate,current_estimate); 
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)	{
    std::fill(gated_gradient[gate_num].begin_all(),
              gated_gradient[gate_num].end_all(),
              0.F);
    this->_single_gate_obj_funcs[gate_num].
      compute_sub_gradient_without_penalty_plus_sensitivity(gated_gradient[gate_num], 
                                                            gated_image_estimate[gate_num], 
                                                            subset_num);
  }	
  //	if(this->_motion_correction_type==-1)
  this->_reverse_motion_vectors.warp_image(gradient,gated_gradient) ; 
  //	else
  //		  this->_motion_vectors.warp_image(gradient,gated_gradient) ; 
}

template<typename TargetT>
double
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
						  const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  double result = 0.;
  GatedDiscretisedDensity gated_image_estimate=this->_gated_image_template;
  // The following initialization doesn't stabilize reconstruction.
  for(unsigned int gate_num=1; gate_num<=this->get_time_gate_definitions().get_num_gates(); ++gate_num)
    std::fill(gated_image_estimate[gate_num].begin_all(),
	      gated_image_estimate[gate_num].end_all(),
	      0.F);
  this->_motion_vectors.warp_image(gated_image_estimate,current_estimate) ;  
  // loop over single_gate
  for(unsigned int gate_num=1 ;
      gate_num<=this->get_time_gate_definitions().get_num_gates();
      ++gate_num)
    {
      result +=	this->_single_gate_obj_funcs[gate_num].
        compute_objective_function_without_penalty(gated_image_estimate[gate_num], 
						   subset_num);
    }
  return result;
}

template<typename TargetT>
void
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
  GatedDiscretisedDensity gated_subset_sensitivity=this->_gated_image_template;

  // loop over single_gate to get original subset sensitivity
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
    {
      const shared_ptr<DiscretisedDensity<3,float> > single_gate_subsensitivity_sptr(
                                                                                     this->_single_gate_obj_funcs[gate_num].get_subset_sensitivity(subset_num).clone());
      gated_subset_sensitivity.set_density_sptr(single_gate_subsensitivity_sptr,gate_num);
    }

  // perform warp
  this->_reverse_motion_vectors.accumulate_warp_image(sensitivity,gated_subset_sensitivity) ; 
}
//! /todo The PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::actual_add_multiplication_with_approximate_sub_Hessian_without_penalty is not validated and at the moment OSSPS does not converge with motion correction.
template<typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
								       const TargetT& input,
								       const int subset_num) const
{
  // TODO this does not add but replace
  {
    std::string explanation;
    if (!input.has_same_characteristics(this->get_subset_sensitivity(0),  ////////////////////
					explanation))
      {
	warning("PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion:\n"
		"sensitivity and input for add_multiplication_with_approximate_Hessian_without_penalty\n"
		"should have the same characteristics.\n%s",
		explanation.c_str());
	return Succeeded::no;
      }
  }   
  GatedDiscretisedDensity gated_input=this->_gated_image_template;
  GatedDiscretisedDensity gated_output=this->_gated_image_template;
  this->_motion_vectors.warp_image(gated_input,input) ;  

  VectorWithOffset<float> scale_factor(1,this->get_time_gate_definitions().get_num_gates());
  for(unsigned int gate_num=1;
      gate_num<=this->get_time_gate_definitions().get_num_gates();
      ++gate_num)
    {
      scale_factor[gate_num]=gated_input[gate_num].find_max();
      /*! /note This is used to avoid higher values than these set in the precompute_denominator_of_conditioner_without_penalty() function. 
        /sa for more information see the recon_array_functions.cxx and the value of the max_quotient (originaly set to 10000.F) */
      gated_input[gate_num]/=scale_factor[gate_num]; 
      this->_single_gate_obj_funcs[gate_num].
	add_multiplication_with_approximate_sub_Hessian_without_penalty(gated_output[gate_num],
									gated_input[gate_num],
									subset_num);      
      gated_output[gate_num]*=scale_factor[gate_num];
    } // end of loop over gates
  this->_reverse_motion_vectors.warp_image(output,gated_output);  
  output/=this->get_time_gate_definitions().get_num_gates(); //Normalizing to get the average value to test if OSSPS works.
  return Succeeded::yes;
}

END_NAMESPACE_STIR
