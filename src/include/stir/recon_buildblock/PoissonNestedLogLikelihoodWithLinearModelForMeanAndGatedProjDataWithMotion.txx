/*
 Copyright (C) 2006- 2009, Hammersmith Imanet Ltd
 Copyright (C) 2011 - 2013, King's College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
 */
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Implementation of class stir::PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion

  \author Nicolas A Karakatsanis

*/
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/numerics/divide.h"
#include "stir/thresholding.h"
#include "stir/NumericInfo.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/RelatedViewgrams.h"
#include "stir/stream.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/CPUTimer.h"

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
#include "stir/recon_buildblock/PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion.h"
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

const float small_num = 0.000001F;

template<typename TargetT>
const char * const 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
registered_name = 
"PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion";

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
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
  
  //Number of nested iterations
  this->num_nested_subiterations=1;
  
  this->maximum_nested_relative_change = NumericInfo<float>().max_value();
  this->minimum_nested_relative_change = 0;
  
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion Parameters");
  this->parser.add_stop_key("End PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion Parameters");
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
    
  // Nested subiterations
  this->parser.add_key("number of nested subiterations",  &this->num_nested_subiterations);
	
  //max and min allowed relative change	between nested updates
  this->parser.add_key("maximum nested relative change", &this->maximum_nested_relative_change);
  this->parser.add_key("minimum nested relative change",&this->minimum_nested_relative_change);
  
}

template<typename TargetT>
bool
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
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
      std::cerr << "\nReading additive projdata data "
                << this->_additive_gated_proj_data_filename 
                << std::endl;
      this->_additive_gated_proj_data_sptr.reset(
                                                 GatedProjData::read_from_file(this->_additive_gated_proj_data_filename));
    }
  if (this->_normalisation_gated_proj_data_filename != "1")
    {
      std::cerr << "\nReading normalisation projdata data "
                << this->_normalisation_gated_proj_data_filename 
                << std::endl;
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
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion()
{
  this->set_defaults();
}

template <typename TargetT>
TargetT *
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
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
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{  // call actual_subsets_are_approximately_balanced() for first single_gate_obj_func
  if (this->get_time_gate_definitions().get_num_gates() == 0 || this->_single_gate_obj_funcs.size() == 0)
    error("PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion:\n"
	  "actual_subsets_are_approximately_balanced called but no gates yet.");
  else if(this->_single_gate_obj_funcs.size() != 0)
    {
      bool gates_are_balanced=true;
      for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
        gates_are_balanced &= this->_single_gate_obj_funcs[gate_num].subsets_are_approximately_balanced(warning_message);
      return gates_are_balanced;
    }
  else 
    error("Something strange happened in PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion:\n"
            "actual_subsets_are_approximately_balanced called before setup()?");
  return 
    false;    
}

/***************************************************************
  get_ functions
***************************************************************/
template <typename TargetT>
const TimeGateDefinitions &
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_time_gate_definitions() const
{	return this->_time_gate_definitions; }

template <typename TargetT>
const GatedProjData& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_gated_proj_data() const
{ return *this->_gated_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<GatedProjData>& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_gated_proj_data_sptr() const
{ return this->_gated_proj_data_sptr; }

template <typename TargetT>
const int 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_max_segment_num_to_process() const
{ return this->_max_segment_num_to_process; }

template <typename TargetT>
const bool 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_zero_seg0_end_planes() const
{ return this->_zero_seg0_end_planes; }

template <typename TargetT>
const GatedProjData& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_additive_gated_proj_data() const
{ return *this->_additive_gated_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<GatedProjData>& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_additive_gated_proj_data_sptr() const
{ return this->_additive_gated_proj_data_sptr; }

template <typename TargetT>
const GatedProjData& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_normalisation_gated_proj_data() const
{ return *this->_normalisation_gated_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<GatedProjData>& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_normalisation_gated_proj_data_sptr() const
{ return this->_normalisation_gated_proj_data_sptr; }

template <typename TargetT>
const ProjectorByBinPair& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_projector_pair() const
{ return *this->_projector_pair_ptr; }

template <typename TargetT>
const shared_ptr<ProjectorByBinPair>& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_projector_pair_sptr() const
{ return this->_projector_pair_ptr; }

template<typename TargetT>
const shared_ptr<TargetT>& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_model_sensitivity_image_sptr() const
{
  return this->model_sensitivity_image_sptr;
}

template<typename TargetT>
const TargetT& 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
get_model_sensitivity_image() const
{
  return *this->model_sensitivity_image_sptr;
}

/***************************************************************
  set_ functions
***************************************************************/
template<typename TargetT>
int
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
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
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
set_time_gate_definitions(const TimeGateDefinitions & time_gate_definitions)
{ this->_time_gate_definitions=time_gate_definitions; }


/***************************************************************
  set_up()
***************************************************************/
template<typename TargetT>
Succeeded 
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
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

    //Computes model sensitivity image by utilizing motion model matrix	
    this->compute_model_sensitivity_image(*target_sptr);
	
    // construct _single_gate_obj_funcs
    this->_single_gate_obj_funcs.resize(1,this->get_time_gate_definitions().get_num_gates());
	   
    for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
      {
        std::cerr << "Objective Function for Gate Number: " << gate_num << "\n";
	this->_single_gate_obj_funcs[gate_num].set_projector_pair_sptr(this->_projector_pair_ptr);
	this->_single_gate_obj_funcs[gate_num].set_proj_data_sptr(this->_gated_proj_data_sptr->get_proj_data_sptr(gate_num));
	this->_single_gate_obj_funcs[gate_num].set_max_segment_num_to_process(this->_max_segment_num_to_process);
	this->_single_gate_obj_funcs[gate_num].set_zero_seg0_end_planes(this->_zero_seg0_end_planes!=0);
	if(this->_additive_gated_proj_data_sptr!=NULL)
	  this->_single_gate_obj_funcs[gate_num].set_additive_proj_data_sptr(this->_additive_gated_proj_data_sptr->get_proj_data_sptr(gate_num));
	this->_single_gate_obj_funcs[gate_num].set_num_subsets(this->num_subsets);
	this->_single_gate_obj_funcs[gate_num].set_frame_num(1);//This should be gate...
        vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
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
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
						                              const TargetT &current_estimate, 
						                              const int subset_num)
{

  // Clone the const TargetT& current estimate to a TargetT nested_estimate   
  shared_ptr<TargetT> current_nested_estimate(current_estimate.get_empty_copy());

  {
    typename TargetT::const_full_iterator current_estimate_iter = current_estimate.begin_all_const(); 
    const typename TargetT::const_full_iterator end_current_estimate_iter = current_estimate.end_all_const(); 
    typename TargetT::full_iterator current_nested_estimate_iter = current_nested_estimate->begin_all(); 
    while (current_estimate_iter!=end_current_estimate_iter) 
    { 
	  *current_nested_estimate_iter = (*current_estimate_iter); 
	  ++current_nested_estimate_iter; ++current_estimate_iter;  
    }
  }
  
  this->compute_nested_sub_gradient_without_penalty_plus_sensitivity(gradient, 
															         *current_nested_estimate, 
															         subset_num);
															   
  this->last_nested_estimate_sptr = current_nested_estimate; 

}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
compute_nested_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
															 TargetT &current_estimate, 
															 const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  GatedDiscretisedDensity gated_gradient=this->_gated_image_template;
  GatedDiscretisedDensity gated_image_estimate=this->_gated_image_template;
  GatedDiscretisedDensity gated_image_reference_data=this->_gated_image_template;
  GatedDiscretisedDensity gated_image_nested_loop_estimate=this->_gated_image_template;  
  GatedDiscretisedDensity gated_sensitivity=this->_gated_image_template;
  
  
  // The following initialization doesn't stabilize reconstruction.
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
    std::fill(gated_image_estimate[gate_num].begin_all(),
	      gated_image_estimate[gate_num].end_all(),
	      0.F);	
		  
  //Print out the min and max values of the initial motion-corrected estimate
  const float current_min_estimate =
	*std::min_element(current_estimate.begin_all(),
					  current_estimate.end_all()); 
  const float current_max_estimate = 
	*std::max_element(current_estimate.begin_all(),
					  current_estimate.end_all());
  cerr << "Initial motion-corrected image estimate " 
	   << ", (min, max): (" << current_min_estimate << ", " << current_max_estimate << ")" << endl;
	
  // Translate (contaminate-with-motion or equivalent to forward project operation for motion model) 
  // from motion-less space to motion-gated space using a motion-corrected image estimate as an input	
  this->_motion_vectors.warp_image(gated_image_estimate,current_estimate);

  //Storage of the current gated images so that to be use as a reference for all nested sub-iterations of the current global sub-iteration  
  gated_image_reference_data = gated_image_estimate;
  
  CPUTimer outer_loop_timer;
  outer_loop_timer.start();
  
  // loop over all motion gates to calculate the sub-gradient of all motion gates (gated gradients)
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)	
    {
	
	 //Get system sensitivity for each motion gate
	 cerr << "Getting system sub-sensitivity image for gate: " << gate_num << "..." << endl;
	 gated_sensitivity[gate_num]=this->_single_gate_obj_funcs[gate_num].get_subset_sensitivity(subset_num);
	
	 //Print out the min and max values of the system dynamic sensitivity image
	 const float current_min_system_gated_sensitivity =
	  *std::min_element(gated_sensitivity[gate_num].begin_all(),
					    gated_sensitivity[gate_num].end_all()); 
	 const float current_max_system_gated_sensitivity = 
	  *std::max_element(gated_sensitivity[gate_num].begin_all(),
					    gated_sensitivity[gate_num].end_all());
	 cerr << "System sensitivity image for gate: " << gate_num
	      << ", (min, max): (" << current_min_system_gated_sensitivity << ", " << current_max_system_gated_sensitivity << ")" << endl;	
	
	 //Compute sub-gradient for each frame
	 cerr << "Compute sub-gradient (update image) for gate: " << gate_num << "." << endl;
     std::fill(gated_gradient[gate_num].begin_all(),
               gated_gradient[gate_num].end_all(),
               0.F);
			  
     this->_single_gate_obj_funcs[gate_num].
      compute_sub_gradient_without_penalty_plus_sensitivity(gated_gradient[gate_num], 
                                                            gated_image_estimate[gate_num], 
                                                            subset_num);
															
     //Print out the min and max values of the sub-gradient for each motion gate
	 const float current_min_outer_loop_gradient =
	  *std::min_element(gated_gradient[gate_num].begin_all(),
					    gated_gradient[gate_num].end_all()); 
	 const float current_max_outer_loop_gradient = 
	  *std::max_element(gated_gradient[gate_num].begin_all(),
					    gated_gradient[gate_num].end_all());
	 cerr << "Outer loop dynamic sub-gradient image (gate) " << gate_num 
	      << ", (min, max): (" << current_min_outer_loop_gradient << ", " << current_max_outer_loop_gradient << ")" << endl;	
	
	 // Perform projection matrix sensitivity division and update for the single outer loop iteration 
	  
	 // Devide by system matrix sensitivity
	 cerr << "Divide sub-gradient (update image) by system sub-sensitivity for gate " << gate_num << "." << endl;
	 divide(gated_gradient[gate_num].begin_all(), 
	 	    gated_gradient[gate_num].end_all(),
	 	    gated_sensitivity[gate_num].begin_all(),
		    small_num);

	//Print out the min and max values of the sub-gradient/sensitivity for each motion gate
	const float current_min_outer_loop_gradient_over_sensitivity =
	*std::min_element(gated_gradient[gate_num].begin_all(),
					  gated_gradient[gate_num].end_all()); 
	const float current_max_outer_loop_gradient_over_sensitivity = 
	*std::max_element(gated_gradient[gate_num].begin_all(),
					  gated_gradient[gate_num].end_all());
	cerr << "Outer loop dynamic sub-gradient/sensitivity image (gate) " << gate_num 
	     << ", (min, max): (" << current_min_outer_loop_gradient_over_sensitivity << ", " << current_max_outer_loop_gradient_over_sensitivity << ")" << endl;	
	
	// Update outer loop dynamic image estimate
	cerr << "Update gated estimate " << gate_num << " with the sub-gradient (update image) of gate " << gate_num << "." << endl;
	DiscretisedDensity<3,float>::const_full_iterator gated_gradient_single_frame_iter = gated_gradient[gate_num].begin_all_const(); 
	DiscretisedDensity<3,float>::const_full_iterator end_gated_gradient_single_frame_iter = gated_gradient[gate_num].end_all_const(); 
	DiscretisedDensity<3,float>::full_iterator gated_image_reference_data_single_frame_iter = gated_image_reference_data[gate_num].begin_all(); 
	while (gated_gradient_single_frame_iter!=end_gated_gradient_single_frame_iter) 
	{ 
	  *gated_image_reference_data_single_frame_iter *= (*gated_gradient_single_frame_iter); 
	  ++gated_image_reference_data_single_frame_iter; ++gated_gradient_single_frame_iter; 
	}
	
	//Print out the min and max values of the outer loop updated gated images for each gate
	const float current_min_outer_loop_updated_image =
	*std::min_element(gated_image_reference_data[gate_num].begin_all(),
					  gated_image_reference_data[gate_num].end_all()); 
	const float current_max_outer_loop_updated_image = 
	*std::max_element(gated_image_reference_data[gate_num].begin_all(),
					  gated_image_reference_data[gate_num].end_all());
	cerr << "Outer loop updated image (gate): " << gate_num 
	     << ", (min, max): (" << current_min_outer_loop_updated_image << ", " << current_max_outer_loop_updated_image << ")" << endl;
	
   }
  
  cerr << "Current outer loop computation time: " << outer_loop_timer.value() << endl << endl;

    CPUTimer nested_loop_timer;
  nested_loop_timer.start();
  
  //nested EM loop
  cerr << endl << "Entering nested loop " << endl;
  
  // This is the principal method that iteratively estimates the motion-corrected estimates in a nested EM loop
  this->estimate_nested_loop_parameters_with_model(gradient,
                                                   current_estimate,
                                                   gated_image_estimate,
											       gated_image_reference_data,
											       gated_image_nested_loop_estimate);
  
  cerr << "Total computation time for " <<  this->num_nested_subiterations << " nested iterations: " << nested_loop_timer.value() << endl << endl;
	
}


template<typename TargetT>
void
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
estimate_nested_loop_parameters_with_model(TargetT &gradient,
                                           TargetT &current_estimate,
                                           GatedDiscretisedDensity &gated_image_estimate,
										   GatedDiscretisedDensity &gated_image_reference_data,
										   GatedDiscretisedDensity &gated_image_nested_loop_estimate)											 
{

  //nested EM loop
  cerr << endl << "Entering nested loop (" << this->num_nested_subiterations << " EM sub-iterations for motion modelling and correction)." << endl;

  
  for(nested_subiterations_num=1; nested_subiterations_num<=this->num_nested_subiterations; nested_subiterations_num++)
  {
    
	// Translate (contaminate-with-motion or equivalent to forward project operation for motion model) 
    // from motion-less space to motion-gated space using a motion-corrected image estimate as an input
	this->_motion_vectors.warp_image(gated_image_nested_loop_estimate,current_estimate); 
    gated_image_estimate = gated_image_reference_data;

	// loop over all motion gates
	for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
	  divide(gated_image_estimate[gate_num].begin_all(),
			 gated_image_estimate[gate_num].end_all(),
			 gated_image_nested_loop_estimate[gate_num].begin_all(),
			 small_num);

	// Reversely translate (correct-for-motion or equivalent to back project operation for motion model)
    // from motion-gated space to motion-corrected single-gate space using the gated image estimate as an input	
    this->_reverse_motion_vectors.warp_image(gradient,gated_image_estimate);																		 

	// Perform model sensitivity division and update for all nested iterations	  

	// Devide by motion model sensitivity
	divide(gradient.begin_all(), 
	       gradient.end_all(),
		   this->model_sensitivity_image_sptr->begin_all(),
		   small_num);


	const float current_min_nested_gradient =
	  *std::min_element(gradient.begin_all(),
	 			        gradient.end_all()); 
	const float current_max_nested_gradient = 
        *std::max_element(gradient.begin_all(),
						  gradient.end_all()); 
	const float new_min_nested_gradient = 
	    static_cast<float>(this->minimum_nested_relative_change);
	const float new_max_nested_gradient = 
	    static_cast<float>(this->maximum_nested_relative_change);
	cerr << "Nested iteration: " << nested_subiterations_num 
		 << " sub-gradient(update image) old value (min, max): (" 
		 << current_min_nested_gradient << ", " << current_max_nested_gradient
		 << "), new value (min, max) (" 
		 << max(current_min_nested_gradient, new_min_nested_gradient) << ", " 
		 << min(current_max_nested_gradient, new_max_nested_gradient) << ")" << endl;

	threshold_upper_lower(gradient.begin_all(),
						  gradient.end_all(), 
						  new_min_nested_gradient, new_max_nested_gradient);      


	//Nested updates of image estimates		
	{		  
	  typename TargetT::const_full_iterator gradient_iter = gradient.begin_all_const(); 
	  const typename TargetT::const_full_iterator end_gradient_iter = gradient.end_all_const(); 
	  typename TargetT::full_iterator current_estimate_iter = current_estimate.begin_all(); 
	  while (gradient_iter!=end_gradient_iter) 
	  { 
		*current_estimate_iter *= (*gradient_iter); 
		++current_estimate_iter; ++gradient_iter;  
	  } 
	}
	  
	//Print out the min and max values of the nested updated image for each nested iteration
	const float current_min_nested_updated_image =
	*std::min_element(current_estimate.begin_all(),
					  current_estimate.end_all()); 
	const float current_max_nested_updated_image = 
	*std::max_element(current_estimate.begin_all(),
					  current_estimate.end_all());
	cerr << "Nested iteration: " << nested_subiterations_num 
	     << " Updated image value (min, max) ("
		 << current_min_nested_updated_image << ", " << current_max_nested_updated_image << ")" << endl << endl;
	
  }
  
  cerr << "End of nested reconstruction process of motion-corrected estimates (after " 
       << this->num_nested_subiterations << " nested EM subiterations)" << endl << endl;
	   
}

template<typename TargetT>
double
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
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
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
compute_model_sensitivity_image(TargetT& motion_corrected_image)
{

  shared_ptr<TargetT> motion_corrected_image_sptr(motion_corrected_image.get_empty_copy());
  this->model_sensitivity_image_sptr=motion_corrected_image_sptr;
  
  //Initialize model sensitivity image
  std::fill(this->model_sensitivity_image_sptr->begin_all(),
            this->model_sensitivity_image_sptr->end_all(),
  	        1.F);
  
  GatedDiscretisedDensity gated_image_of_all_ones=this->_gated_image_template;

  // loop over all motion gates
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
      std::fill(gated_image_of_all_ones[gate_num].begin_all(),
	            gated_image_of_all_ones[gate_num].end_all(),
	            1.F);
		
  cerr << "Computing model sensitivity image..." << endl;

  // To obtain motion model sensitivity image reversely translate (correct-for-motion or equivalent to back project operation for motion model)
  // from motion-gated space to motion-corrected single-gate space using a gated image estimate of ALL ONES	
  this->_reverse_motion_vectors.warp_image(*this->model_sensitivity_image_sptr,
                                           gated_image_of_all_ones);					
  
  //Print out the min and max values of the model sensitivity image
  const float current_min_model_sensitivity =
  *std::min_element(this->model_sensitivity_image_sptr->begin_all(),
   				    this->model_sensitivity_image_sptr->end_all()); 
  const float current_max_model_sensitivity = 
  *std::max_element(this->model_sensitivity_image_sptr->begin_all(),
				    this->model_sensitivity_image_sptr->end_all());
  cerr << "Model sensitivity image " 
	   << ", (min, max): (" << current_min_model_sensitivity << ", " << current_max_model_sensitivity << ")" << endl;
  
  cerr << "Model sensitivity image has been computed." << endl; 
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
add_subset_sensitivity(TargetT& model_sensitivity, const int subset_num) const
{
  GatedDiscretisedDensity gated_image_of_all_ones=this->_gated_image_template;

  // loop over all motion gates
  for(unsigned int gate_num=1;gate_num<=this->get_time_gate_definitions().get_num_gates();++gate_num)
      std::fill(gated_image_of_all_ones[gate_num].begin_all(),
	            gated_image_of_all_ones[gate_num].end_all(),
	            1.F);

  // To obtain motion model sensitivity image reversely translate (correct-for-motion or equivalent to back project operation for motion model) 
  // AND accumulate over previous calls the resulting motion corrected images. 
  // A gated image estimate of ALL ONES is used as input at every call of the function	
  this->_reverse_motion_vectors.accumulate_warp_image(model_sensitivity,
                                                      gated_image_of_all_ones);

}


//! /todo The PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::actual_add_multiplication_with_approximate_sub_Hessian_without_penalty is not validated and at the moment OSSPS does not converge with motion correction.
template<typename TargetT>
Succeeded
PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>::
actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
								       const TargetT& input,
								       const int subset_num) const
{
  // TODO this does not add but replace
  {
    string explanation;
    if (!input.has_same_characteristics(this->get_subset_sensitivity(0),  ////////////////////
					explanation))
      {
	warning("PoissonNestedLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion:\n"
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
