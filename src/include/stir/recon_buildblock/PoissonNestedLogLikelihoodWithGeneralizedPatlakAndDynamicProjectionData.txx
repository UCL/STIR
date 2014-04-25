//
/*
  Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
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
  \ingroup modelling
  \brief Implementation of class stir::PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData

  \author Nicolas A Karakatsanis

  $Date: 2013-07-12 10:34:00 $
  $Revision: 1.0 $
*/
#include "stir/DiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"

#include "stir/Array.h"
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

#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"

#include "stir/Viewgram.h"
#include "stir/recon_array_functions.h"
#include <algorithm>
#include <string> 
// For the Patlak Plot Modelling
#include "stir/modelling/GeneralizedPatlakMatrix.h"
#include "stir/modelling/ModelMatrix.h"
#include "stir/recon_buildblock/PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

//const float small_num = 0.000001F;

template<typename TargetT>
const char * const 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
registered_name = 
"PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData";

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
set_defaults()
{
  base_type::set_defaults();

  this->_input_filename="";
  this->_max_segment_num_to_process=-1;
  //num_views_to_add=1;    // KT 20/06/2001 disabled

  this->_dyn_proj_data_sptr.reset();
  this->_zero_seg0_end_planes = 0;

  this->_additive_dyn_proj_data_filename = "0";
  this->_additive_dyn_proj_data_sptr.reset();

#ifndef USE_PMRT // set default for _projector_pair_ptr
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingRayTracing());
  shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingInterpolation());
#else
  shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingProjMatrixByBin(PM)); 
  shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingProjMatrixByBin(PM)); 
#endif

  this->_projector_pair_ptr.
    reset(new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));
  this->_normalisation_sptr.reset(new TrivialBinNormalisation);

  // image stuff
  this->_output_image_size_xy=-1;
  this->_output_image_size_z=-1;
  this->_zoom=1.F;
  this->_Xoffset=0.F;
  this->_Yoffset=0.F;
  this->_Zoffset=0.F;   // KT 20/06/2001 new

  // A counter measuring all the number of global (not nested) iterations (both initialization and regular recosntruction mode).
  this->subiterations_counter=0;
  
  //Number of nested iterations
  this->num_nested_initialization_subiterations=1;
  this->num_nested_subiterations=1;
    
  this->maximum_nested_relative_change = NumericInfo<float>().max_value();
  this->minimum_nested_relative_change = 0;
  
  // Modelling Stuff
  this->_patlak_plot_sptr.reset();  //For kinetic modelling
  
  // Initializing generalized Patlak Ki, kloss and V parameters from standard Patlak Ki and V parameters (kloss is initialized with zero)
  // The following parameter determines the number of complete 4D Standard Patlak iterations required for initialization
  // Default value is 1 (i.e. only perform a single standard Patlak iteration (with as many nested iterations 
  // as defined from "this->this->num_nested_initialization_subiterations" at the first iteration and use generalized Patlak for the subsequent iterations)
  this->num_initialization_subiterations=7;
  
  // A counter measuring the number of global (not nested) iterations executed under kinetic model initialization mode.
  this->initialization_subiterations_counter=0;
  
  // The default choice is only to perform purely standard Patlak nested reconstruction, i.e.
  // NOT to alternate between the standard (first) and generalized (afterwards) Patlak at initialization mode.
  this->is_alternating_initialization_model=0;
  
  //Initializing generalized Patlak Ki, kloss and V parameters with a global value
  //Currently not used, but it was retained for future usage
  //this->global_param_initialization=1;
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData Parameters");
  this->parser.add_stop_key("End PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData Parameters");
  this->parser.add_key("input file",&this->_input_filename);

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
  this->parser.add_key("additive sinograms",&this->_additive_dyn_proj_data_filename);

  // normalisation (and attenuation correction)
  this->parser.add_parsing_key("Bin Normalisation type", &this->_normalisation_sptr);

  // Modelling Information
  this->parser.add_parsing_key("Kinetic Model Type", &this->_patlak_plot_sptr); // Do sth with dynamic_cast to get the GeneralizedPatlakPlot
  
  // Nested subiterations (regular nested iterations using the Generalized Patlak matrix)
  this->parser.add_key("number of nested subiterations",  &this->num_nested_subiterations);
  
  // Regularization Information
  //  this->parser.add_parsing_key("prior type", &this->_prior_sptr);
  
  // Global initial subiterations for initialization (using standard Patlak model, i.e. ModelMatrix) before switching to Generalized Patlak iterations
  this->parser.add_key("number of global initialization subiterations",  &this->num_initialization_subiterations);
	
  // Nested subiterations for initialization (using standard Patlak model, i.e. ModelMatrix instead of GenearlizedPatlakMatrix)
  this->parser.add_key("number of nested initialization subiterations",  &this->num_nested_initialization_subiterations);

  // Choose whether both standard and generalized Patlak models 
  // (1) will be used in an alternating fashion for the initialization or 
  // (0) only the standard Patlak model will be utilized.
  this->parser.add_key("alternating initialization mode",  &this->is_alternating_initialization_model);
	
  //max and min allowed relative change	between nested updates
  this->parser.add_key("maximum nested relative change", &this->maximum_nested_relative_change);
  this->parser.add_key("minimum nested relative change", &this->minimum_nested_relative_change);
  
  //Initializing generalized Patlak parameters Ki, kloss and V with a global value
  //Currently not defined, but it is retained for future usage
  //this->parser.add_key("parameters initialization global value", &this->global_param_initialization);
}

template<typename TargetT>
bool
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  if (this->_input_filename.length() == 0)
    { warning("You need to specify an input filename"); return true; }
  
#if 0 // KT 20/06/2001 disabled as not functional yet
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
    { warning("The 'mash x views' key has an invalid value (must be 1 or even number)"); return true; }
#endif
 
  this->_dyn_proj_data_sptr.reset(DynamicProjData::read_from_file(_input_filename));
  if (is_null_ptr(this->_dyn_proj_data_sptr))
    { warning("Error reading input file %s", _input_filename.c_str()); return true; }
  // image stuff
  if (this->_zoom <= 0)
    { warning("zoom should be positive"); return true; }
  
  if (this->_output_image_size_xy!=-1 && this->_output_image_size_xy<1) // KT 10122001 appended_xy
    { warning("output image size xy must be positive (or -1 as default)"); return true; }
  if (this->_output_image_size_z!=-1 && this->_output_image_size_z<1) // KT 10122001 new
    { warning("output image size z must be positive (or -1 as default)"); return true; }


  if (this->_additive_dyn_proj_data_filename != "0")
    {
      cerr << "\nReading additive projdata data "
           << this->_additive_dyn_proj_data_filename 
           << std::endl;
      this->_additive_dyn_proj_data_sptr.reset(DynamicProjData::read_from_file(this->_additive_dyn_proj_data_filename));
      if (is_null_ptr(this->_additive_dyn_proj_data_sptr))
	{ warning("Error reading additive input file %s", _additive_dyn_proj_data_filename.c_str()); return true; }

    }
  return false;
}

template <typename TargetT>
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData()
{
  this->set_defaults();
}

template <typename TargetT>
TargetT *
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
construct_target_ptr() const
{  
  return
    new GeneralizedPatlakVoxelsOnCartesianGrid(GeneralizedPatlakVoxelsOnCartesianGridBaseType(
                                                                                *(this->_dyn_proj_data_sptr->get_proj_data_info_ptr()),
                                                                                static_cast<float>(this->_zoom),
                                                                                CartesianCoordinate3D<float>(static_cast<float>(this->_Zoffset),
                                                                                                             static_cast<float>(this->_Yoffset),
                                                                                                             static_cast<float>(this->_Xoffset)),
                                                                                CartesianCoordinate3D<int>(this->_output_image_size_z,
                                                                                                           this->_output_image_size_xy,
                                                                                                           this->_output_image_size_xy)));
}
/***************************************************************
  subset balancing
***************************************************************/

template<typename TargetT>
bool
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
actual_subsets_are_approximately_balanced(std::string& warning_message) const
{  // call actual_subsets_are_approximately_balanced( for first single_frame_obj_func )
  if (this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames() == 0 || this->_single_frame_obj_funcs.size() == 0)
    error("PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData:\n"
          "actual_subsets_are_approximately_balanced called but not frames yet.\n");
  else if(this->_single_frame_obj_funcs.size() != 0)
    {
      bool frames_are_balanced=true;
      for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
        frames_are_balanced &= this->_single_frame_obj_funcs[frame_num].subsets_are_approximately_balanced(warning_message);
      return frames_are_balanced;
    }
  else 
    warning("Something strange happened in PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData:\n"
            "actual_subsets_are_approximately_balanced called before setup()?\n");
  return 
    false;    
}

/***************************************************************
  get_ functions
***************************************************************/
template <typename TargetT>
const DynamicProjData& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_dyn_proj_data() const
{ return *this->_dyn_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<DynamicProjData>& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_dyn_proj_data_sptr() const
{ return this->_dyn_proj_data_sptr; }

template <typename TargetT>
const int 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_max_segment_num_to_process() const
{ return this->_max_segment_num_to_process; }

template <typename TargetT>
const bool 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_zero_seg0_end_planes() const
{ return this->_zero_seg0_end_planes; }

template <typename TargetT>
const DynamicProjData& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_additive_dyn_proj_data() const
{ return *this->_additive_dyn_proj_data_sptr; }

template <typename TargetT>
const shared_ptr<DynamicProjData>& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_additive_dyn_proj_data_sptr() const
{ return this->_additive_dyn_proj_data_sptr; }

template <typename TargetT>
const ProjectorByBinPair& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_projector_pair() const
{ return *this->_projector_pair_ptr; }

template <typename TargetT>
const shared_ptr<ProjectorByBinPair>& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_projector_pair_sptr() const
{ return this->_projector_pair_ptr; }

template <typename TargetT>
const BinNormalisation& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_normalisation() const
{ return *this->_normalisation_sptr; }

template <typename TargetT>
const shared_ptr<BinNormalisation>& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_normalisation_sptr() const
{ return this->_normalisation_sptr; }


template<typename TargetT>
const DynamicDiscretisedDensity 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_model_sensitivity_impulse_response() const
{
  return this->sensitivity_impulse_response_image;
}

template<typename TargetT>
const TargetT& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_initialization_model_sensitivity_image() const
{
  return *this->initialization_model_sensitivity_image_sptr;
}

template<typename TargetT>
const shared_ptr<TargetT>& 
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
get_initialization_model_sensitivity_image_sptr() const
{
  return this->initialization_model_sensitivity_image_sptr;
}


/***************************************************************
  set_ functions
***************************************************************/
template<typename TargetT>
int
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
set_num_subsets(const int num_subsets)
{
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
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
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
set_up_before_sensitivity(shared_ptr<TargetT > const& target_sptr)
{
  if (this->_max_segment_num_to_process==-1)
    this->_max_segment_num_to_process =
      (this->_dyn_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num();

  if (this->_max_segment_num_to_process > (this->_dyn_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num()) 
    { 
      warning("_max_segment_num_to_process (%d) is too large",
              this->_max_segment_num_to_process); 
      return Succeeded::no;
    }

  shared_ptr<ProjDataInfo> proj_data_info_sptr(
					       (this->_dyn_proj_data_sptr->get_proj_data_sptr(1))->get_proj_data_info_ptr()->clone());
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
 
	
  if (this->_patlak_plot_sptr->set_up() == Succeeded::no)
    {
	  cerr << "Generalized Patlak Plot set up did not succeed!" << endl;
	  return Succeeded::no;
	}

  if (this->_patlak_plot_sptr->get_starting_frame()<=0 || this->_patlak_plot_sptr->get_starting_frame()>this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames())
    {
      warning("Starting frame is %d. Generally, it should be a late frame,\nbut in any case it should be less than the number of frames %d\nand at least 1.",this->_patlak_plot_sptr->get_starting_frame(), this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames());
      return Succeeded::no;
    }	
	
  {
    const shared_ptr<DiscretisedDensity<3,float> > 
      density_template_sptr((target_sptr->construct_single_density(1)).get_empty_copy());
    const shared_ptr<Scanner> scanner_sptr(new Scanner(*proj_data_info_sptr->get_scanner_ptr()));
    this->_dyn_image_template=
      DynamicDiscretisedDensity(this->_patlak_plot_sptr->get_time_frame_definitions(), 
                                this->_dyn_proj_data_sptr->get_start_time_in_secs_since_1970(),
                                scanner_sptr,
                                density_template_sptr);
	this->_imp_response_image_template=
      DynamicDiscretisedDensity(this->_patlak_plot_sptr->get_time_frame_definitions(), 
                                this->_patlak_plot_sptr->get_num_conv_params(),
								this->_dyn_proj_data_sptr->get_start_time_in_secs_since_1970(),
                                scanner_sptr,
                                density_template_sptr);

	//Initialize an impulse reponse vector according to the GeneralizedPatlak plot model matrix	
    if(!((this->_patlak_plot_sptr->get_model_matrix()).get_model_array()).get_regular_range(this->model_array_min,this->model_array_max))
      error("Model array has not regular range");

	this->impulse_response=this->_imp_response_image_template;
	this->sensitivity_impulse_response_image=this->_imp_response_image_template;
	
	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num) 
      std::fill(this->impulse_response[conv_param_num].begin_all(),
                this->impulse_response[conv_param_num].end_all(),
                1.F);
				
	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num) 
      std::fill(this->sensitivity_impulse_response_image[conv_param_num].begin_all(),
                this->sensitivity_impulse_response_image[conv_param_num].end_all(),
                1.F);
	
	//Computes model sensitivity image by utilizing GeneralizedPatlak plot model matrix	
    this->compute_model_sensitivity_impulse_response(this->sensitivity_impulse_response_image);
	
    //Computes initialization model sensitivity image by utilizing the Generalized Patlak plot model matrix	initialization methods
    this->compute_initialization_model_sensitivity_image(*target_sptr);
	  
	//this->impulse_response =  VectorWithOffset<float>(this->model_array_min[1],this->model_array_max[1]);
	//this->sensitivity_impulse_response_vector =  VectorWithOffset<float>(this->model_array_min[1],this->model_array_max[1]);
	
	//By default during set-up we start the first iteration with initialization mode. 
	//However, if user selects "this->num_initialization_subiterations=0" at the par file, then no initialization mode is activated at all.
    this->is_initialization_subiteration=true;
	
    // construct _single_frame_obj_funcs
    this->_single_frame_obj_funcs.resize(this->_patlak_plot_sptr->get_starting_frame(),this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames());
   
    for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
      {
        this->_single_frame_obj_funcs[frame_num].set_projector_pair_sptr(this->_projector_pair_ptr);
        this->_single_frame_obj_funcs[frame_num].set_proj_data_sptr(this->_dyn_proj_data_sptr->get_proj_data_sptr(frame_num));
        this->_single_frame_obj_funcs[frame_num].set_max_segment_num_to_process(this->_max_segment_num_to_process);
        this->_single_frame_obj_funcs[frame_num].set_zero_seg0_end_planes(this->_zero_seg0_end_planes!=0);
        if(this->_additive_dyn_proj_data_sptr!=NULL)
          this->_single_frame_obj_funcs[frame_num].set_additive_proj_data_sptr(this->_additive_dyn_proj_data_sptr->get_proj_data_sptr(frame_num));
        this->_single_frame_obj_funcs[frame_num].set_num_subsets(this->num_subsets);
        this->_single_frame_obj_funcs[frame_num].set_frame_num(frame_num);
        this->_single_frame_obj_funcs[frame_num].set_frame_definitions(this->_patlak_plot_sptr->get_time_frame_definitions());
        this->_single_frame_obj_funcs[frame_num].set_normalisation_sptr(this->_normalisation_sptr);
        this->_single_frame_obj_funcs[frame_num].set_recompute_sensitivity(this->get_recompute_sensitivity());
        this->_single_frame_obj_funcs[frame_num].set_use_subset_sensitivities(this->get_use_subset_sensitivities());
        if(this->_single_frame_obj_funcs[frame_num].set_up(density_template_sptr) != Succeeded::yes)
          error("Single frame objective functions is not set correctly!");
      }
  }//_single_frame_obj_funcs[frame_num]

  return Succeeded::yes;
}

/*************************************************************************
  functions that compute the value/gradient of the objective function etc
*************************************************************************/

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
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
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
compute_nested_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
															 TargetT &current_estimate, 
															 const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  DynamicDiscretisedDensity dyn_gradient=this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_image_estimate=this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_image_reference_data=this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_image_nested_loop_estimate=this->_dyn_image_template;  
  DynamicDiscretisedDensity dyn_sensitivity=this->_dyn_image_template;

  DynamicDiscretisedDensity impulse_response_gradient=this->_imp_response_image_template;

  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
    std::fill(dyn_image_estimate[frame_num].begin_all(),
              dyn_image_estimate[frame_num].end_all(),
              1.F);

  for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num) 
    std::fill(impulse_response_gradient[conv_param_num].begin_all(),
              impulse_response_gradient[conv_param_num].end_all(),
              1.F);

  //The counter below measures all the global subiterations (both the initialization and the regular ones)
  this->subiterations_counter++;
			  
  // Only under initialization mode, i.e. when this->is_initialization_subiteration==true, initialize parametric image estimate with standard Patlak
  // By default during set-up we start the first iteration with initialization mode and 
  // continue for as many iterations as determined by user-defined parameter: "this->num_initialization_subiterations". 
  // However, if user selects "this->num_initialization_subiterations=0" at the par file, then no initialization mode is activated at all.
  if (this->is_initialization_subiteration)
    {
	  this->initialization_subiterations_counter++;
	  
	  if (this->initialization_subiterations_counter>this->num_initialization_subiterations)
	    this->is_initialization_subiteration=false;
	}
	
  //Print out the min and max values of the initial parametric estimate after initialization
  const float current_min_estimate =
	*std::min_element(current_estimate.begin_all(),
					  current_estimate.end_all()); 
  const float current_max_estimate = 
	*std::max_element(current_estimate.begin_all(),
		   			  current_estimate.end_all());
  cerr << "Initial parametric image " 
	     << ", (min, max): (" << current_min_estimate << ", " << current_max_estimate << ")" << endl;

  if (this->is_initialization_subiteration)
    {
      //At initialization mode, standard Patlak model is used and, therefore, dynamic image can be directly obtained from parametric image estimate.
	  this->_patlak_plot_sptr->get_dynamic_image_from_initialization_parametric_image(dyn_image_estimate,
	                                                                                  current_estimate); 
    }
  else
    {  
      
	  //At regular mode, we can indirectly (through imp response) forward project from parameter space to time space using a parametric image estimate as an input
      //Not used any more, since the indirect fwd proj was broken down to two dicrete steps: parametric image->imp response image->dynamic image  
      //this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_image_estimate,current_estimate);
	  
	  //At regular mode, synthesize initial impulse response image from previous full iteration parametric image estimate	 
      this->_patlak_plot_sptr->get_impulse_response_from_parametric_image(this->impulse_response,current_estimate);
  
      //Print out the min and max values of initial impulse response (only for first and last convolution time point)
      cerr << " Initial impulse response current value for current full iteration [conv point](min, max):" << endl;
	
      for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	    {
	     if ((conv_param_num==model_array_min[1]) || (conv_param_num==model_array_max[1]))
		   {
		     const float current_min_initial_impulse_response =
	          *std::min_element(this->impulse_response[conv_param_num].begin_all(),
		  				        this->impulse_response[conv_param_num].end_all()); 
	         const float current_max_initial_impulse_response = 
              *std::max_element(this->impulse_response[conv_param_num].begin_all(),
						        this->impulse_response[conv_param_num].end_all());
	  
	         cerr << "	[" << conv_param_num << "](" << current_min_initial_impulse_response << ", " << current_max_initial_impulse_response << ")		" << endl;
			}
	    }
      cerr << endl;
  
      //Then convolve the initial impulse response with the input function matrix to get the dynamic images (2nd step of kinetic forward projection)
      this->_patlak_plot_sptr->get_dynamic_image_from_impulse_response(dyn_image_estimate,this->impulse_response);
  
    }
	
  dyn_image_reference_data = dyn_image_estimate;
  
  CPUTimer outer_loop_timer;
  outer_loop_timer.start();
  
  // loop over single_frame and use model_matrix
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
    {
    //Printing out the min and max values of each frame of the forward projected dynamic image (current dyn frame)
    const float min_dyn_image_estimate =
      *std::min_element(dyn_image_estimate[frame_num].begin_all(),
				        dyn_image_estimate[frame_num].end_all()); 
    const float max_dyn_image_estimate = 
      *std::max_element(dyn_image_estimate[frame_num].begin_all(),
				        dyn_image_estimate[frame_num].end_all());
    cerr << "Forward projected dynamic image outer loop estimate for frame: " << frame_num << " (min, max): (" 
         << min_dyn_image_estimate << ", " << max_dyn_image_estimate
         << ") " << endl;
  
    //Get system sensitivity for each dynamic frame
	cerr << "Getting system sub-sensitivity image for dynamic frame: " << frame_num << "..." << endl;
	dyn_sensitivity[frame_num]=this->_single_frame_obj_funcs[frame_num].get_subset_sensitivity(subset_num);
	
	 //Print out the min and max values of the system dynamic sensitivity image
	const float current_min_system_dyn_sensitivity =
	*std::min_element(dyn_sensitivity[frame_num].begin_all(),
					  dyn_sensitivity[frame_num].end_all()); 
	const float current_max_system_dyn_sensitivity = 
	*std::max_element(dyn_sensitivity[frame_num].begin_all(),
					  dyn_sensitivity[frame_num].end_all());
	cerr << "System sensitivity image for dynamic frame: " << frame_num
	     << ", (min, max): (" << current_min_system_dyn_sensitivity << ", " << current_max_system_dyn_sensitivity << ")" << endl;	
	
	//Compute sub-gradient for each frame
	cerr << "Compute sub-gradient (update image) for dynamic frame: " << frame_num << "." << endl;
	std::fill(dyn_gradient[frame_num].begin_all(),
			dyn_gradient[frame_num].end_all(),
			1.F);

	this->_single_frame_obj_funcs[frame_num].
	  compute_sub_gradient_without_penalty_plus_sensitivity(dyn_gradient[frame_num], 
														  dyn_image_estimate[frame_num], 
														  subset_num);
															  
	//Print out the min and max values of the sub-gradient for each fynamic frame
	const float current_min_outer_loop_gradient =
	*std::min_element(dyn_gradient[frame_num].begin_all(),
					  dyn_gradient[frame_num].end_all()); 
	const float current_max_outer_loop_gradient = 
	*std::max_element(dyn_gradient[frame_num].begin_all(),
					  dyn_gradient[frame_num].end_all());
	cerr << "Outer loop dynamic sub-gradient image (frame " << frame_num 
	     << "), (min, max): (" << current_min_outer_loop_gradient << ", " << current_max_outer_loop_gradient << ")" << endl;	
	
	// Perform projection matrix sensitivity division and update for the single outer loop iteration 
	  
	// Devide by system matrix sensitivity
	cerr << "Divide sub-gradient (update image) by system sub-sensitivity for dynamic frame " << frame_num << "." << endl;
	divide(dyn_gradient[frame_num].begin_all(), 
		   dyn_gradient[frame_num].end_all(),
		   dyn_sensitivity[frame_num].begin_all(),
		   small_num);

	//Print out the min and max values of the sub-gradient/sensitivity for each fynamic frame
	const float current_min_outer_loop_gradient_over_sensitivity =
	*std::min_element(dyn_gradient[frame_num].begin_all(),
					  dyn_gradient[frame_num].end_all()); 
	const float current_max_outer_loop_gradient_over_sensitivity = 
	*std::max_element(dyn_gradient[frame_num].begin_all(),
					  dyn_gradient[frame_num].end_all());
	cerr << "Outer loop dynamic sub-gradient/sensitivity image (frame " << frame_num 
	     << "), (min, max): (" << current_min_outer_loop_gradient_over_sensitivity << ", " << current_max_outer_loop_gradient_over_sensitivity << ")" << endl;	
	
	// Update outer loop dynamic image estimate
	cerr << "Update dynamic image frame estimate " << frame_num << " with the sub-gradient (update image) of dynamic frame " << frame_num << "." << endl;
	DiscretisedDensity<3,float>::const_full_iterator dyn_gradient_single_frame_iter = dyn_gradient[frame_num].begin_all_const(); 
	DiscretisedDensity<3,float>::const_full_iterator end_dyn_gradient_single_frame_iter = dyn_gradient[frame_num].end_all_const(); 
	DiscretisedDensity<3,float>::full_iterator dyn_image_reference_data_single_frame_iter = dyn_image_reference_data[frame_num].begin_all(); 
	while (dyn_gradient_single_frame_iter!=end_dyn_gradient_single_frame_iter) 
	{ 
	  *dyn_image_reference_data_single_frame_iter *= (*dyn_gradient_single_frame_iter); 
	  ++dyn_image_reference_data_single_frame_iter; ++dyn_gradient_single_frame_iter; 
	}
	
	//Print out the min and max values of the outer loop updated dynamic images for each fynamic frame
	const float current_min_outer_loop_updated_image =
	*std::min_element(dyn_image_reference_data[frame_num].begin_all(),
					  dyn_image_reference_data[frame_num].end_all()); 
	const float current_max_outer_loop_updated_image = 
	*std::max_element(dyn_image_reference_data[frame_num].begin_all(),
					  dyn_image_reference_data[frame_num].end_all());
	cerr << "Outer loop updated image (frame): " << frame_num 
	     << ", (min, max): (" << current_min_outer_loop_updated_image << ", " << current_max_outer_loop_updated_image << ")" << endl << endl << endl;
	
  }
  
  cerr << "Current outer loop computation time: " << outer_loop_timer.value() << endl << endl;

  
  if (this->is_initialization_subiteration)
    {
      //ITERATIVE NESTED INITIALIZATION MODE
	  
	  if (this->initialization_subiterations_counter==1)
	    {
	      cerr << endl << endl << "ENTERING 4D RECONSTRUCTION INITIALIZATION MODE." << endl << endl;
		  cerr << "NOTE regarding kloss initialization:" << endl
		       << "User has opted for model initialization with standard Patlak model, i.e. kloss is initialized with ZEROS regardless of users initial kloss estimate" << endl
			   << "If user wishes to initialize kloss with their own NON-ZERO kloss value (not recommended, unless they have a pretty good estimate of true kloss), then they should BOTH: " << endl
			   << "1) Deactivate initialization mode (not recommended) AND " << endl
			   << "2) Specify their own initial parametric image estimates (exercise with caution)." << endl
			   << "In any other case, kloss image will be initialized with ZEROS (recommended), before regular 4D reconstruction mode. " << endl << endl;
		  
		  if (this->is_alternating_initialization_model)
			  cerr << "ALTERNATING INITIALIZATION MODE IS ON." << endl 
			       << "After " << this->num_nested_initialization_subiterations << " initialization standard Patlak nested EM iterations, " 
				   << this->num_nested_subiterations << " initialization generalized Patlak nested EM iterations will follow, within each full initialization iteration." << endl << endl
                   << "Please NOTE that the number of nested generalized Patlak iterations utilized within each initialization full iteration are always equal to " << endl
				   << "the number of nested generalized Patlak iterations selected by the user for the regular full iterations" << endl << endl;
		  else
			  cerr << "ALTERNATING INITIALIZATION MODE IS OFF." << endl << endl;  
	    }
	  
	  //Only at initialization mode perform standard Patlak nested loop updates of the parametric image estimates. 
      
	  CPUTimer nested_initialization_loop_timer;
      nested_initialization_loop_timer.start();
  
      //Entering nested EM initialization loop
  
      // This method iteratively estimates standard Patlak estimates in a nested initialization EM loop to properly 
      // initialize the estimates passed to the the next nested loop which performs the regular generalized Patlak reconstruction
      this->initialize_nested_loop_parameters_with_initialization_model(gradient,
                                                                        current_estimate,
                                                                        dyn_image_estimate,
											                            dyn_image_reference_data,
											                            dyn_image_nested_loop_estimate);
  
      cerr << "Total computation time for " <<  this->num_nested_initialization_subiterations 
           << " initialization standard Patlak nested EM initialization iterations: " << nested_initialization_loop_timer.value() << endl << endl;
	
      //  If alternating initialization is activated, also perform generalized Patlak nested loop updates 
	  //  after the standard Patlak updates, at each global initialization iteration. 	
	  if (this->is_alternating_initialization_model)
	    {
	       CPUTimer nested_loop_timer;
           nested_loop_timer.start();
		   
		   cerr << "Switching from standard Patlak initialization to generalized Patlak initialization iterations. " << endl << endl;

           // This is the principal method that iteratively estimates the generalized Patlak estimates in a nested EM loop 
		   // At this section of the code, nested EM generalized Patlak updates are alternating, for initialization purposes, with nested EM standard Patlak updates
           this->estimate_nested_loop_parameters_with_model(impulse_response_gradient,
		                                                    current_estimate,
                                                            dyn_image_estimate,
                                                            dyn_image_reference_data,
                                                            dyn_image_nested_loop_estimate);

           cerr << "Total computation time for " <<  this->num_nested_subiterations 
	            << " generalized Patlak nested EM initialization iterations: " << nested_loop_timer.value() << endl << endl;
	    }
	  
    }
  else
    {
	  //ITERATIVE NESTED REGULAR RECONSTRUCTION MODE
	  
	  if (this->subiterations_counter==this->num_initialization_subiterations+1)
	    cerr << endl << endl << "ENTERING REGULAR 4D RECONSTRUCTION MODE." << endl << endl
		     << "Only generalized Patlak nested EM updates are performed onwards." << endl << endl
			 << "NOTE regarding kloss initialization: " << endl
			 << "Unless user has opted for: " << endl
			 << "1) for NO model initialization (not recommended) AND " << endl
			 << "2) their OWN NON-ZERO kloss initial estimate (exercise caution), then" << endl
			 << "kloss image is initialized with ZEROS (recommended), by default, before the first nested EM generalized Patlak loop." << endl << endl;
	  
      CPUTimer nested_loop_timer;
      nested_loop_timer.start();

      // This is the principal method that iteratively estimates the generalized Patlak estimates in a nested EM loop
      this->estimate_nested_loop_parameters_with_model(impulse_response_gradient,
	                                                   current_estimate,
                                                       dyn_image_estimate,
                                                       dyn_image_reference_data,
                                                       dyn_image_nested_loop_estimate);

      cerr << "Total computation time for " <<  this->num_nested_subiterations 
	       << " generalized Patlak nested EM iterations: " << nested_loop_timer.value() << endl << endl;
	}
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
estimate_nested_loop_parameters_with_model(DynamicDiscretisedDensity &impulse_response_gradient,
                                           TargetT &current_estimate,
                                           DynamicDiscretisedDensity &dyn_image_estimate,
										   DynamicDiscretisedDensity &dyn_image_reference_data,
										   DynamicDiscretisedDensity &dyn_image_nested_loop_estimate)											 
{

  // Now synthesize the initial impulse response before the first nested generalized Patlak iteration 
  // using as initial estimates the last iteration's parametric image estimates (Initialization step of the nested EM loop)
  cerr << endl << "Synthesizing the initial impulse response from the previous generalized Patlak parameter estimates ..." << endl << endl;
  this->_patlak_plot_sptr->get_impulse_response_from_parametric_image(this->impulse_response,current_estimate);
  
  //Print out the min and max values of initial impulse response (only for first and last convolution time point)
  cerr << "Initialized impulse response value [conv point](min, max):" << endl;
	
  for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	  {
	  	if ((conv_param_num==model_array_min[1]) || (conv_param_num==model_array_max[1]))
	      {
	        const float current_min_nested_initial_impulse_response =
	          *std::min_element(this->impulse_response[conv_param_num].begin_all(),
						        this->impulse_response[conv_param_num].end_all()); 
	        const float current_max_nested_initial_impulse_response = 
              *std::max_element(this->impulse_response[conv_param_num].begin_all(),
						        this->impulse_response[conv_param_num].end_all());
	  
	        cerr << "	[" << conv_param_num << "](" << current_min_nested_initial_impulse_response << ", " << current_max_nested_initial_impulse_response << ")		" << endl;
		  }
	  }
  cerr << endl;
  
  //nested EM loop
  cerr << endl << "Entering nested loop (" << this->num_nested_subiterations << " generalized Patlak EM subiterations)." << endl;
  
  for(nested_subiterations_num=1;nested_subiterations_num<=this->num_nested_subiterations; nested_subiterations_num++)
  {
 
    //Print out the min and max values of the initial parametric estimate at the beginning of each nested update
	const float current_min_estimate_nested =
	*std::min_element(current_estimate.begin_all(),
					  current_estimate.end_all()); 
	const float current_max_estimate_nested = 
	*std::max_element(current_estimate.begin_all(),
					  current_estimate.end_all());
	cerr << "Nested iteration: " << nested_subiterations_num
	     << " Initial parametric image for the current nested iteration " 
	     << ", (min, max): (" << current_min_estimate_nested << ", " << current_max_estimate_nested << ")" << endl;

	
    // This function is used to directly obtain dynamic images from the parametric image estimates. 
	// Not used anymore, as the process has been broken down to two steps: parametric image->impulse response image->dynamic image	
	//this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_image_nested_loop_estimate,current_estimate) ; 
 
    // First synthesize the initial impulse response at each nested iteration from the current parametric image estimate (1st step of kinetic forward projection)
	// Not used anymore, as the impulse response image is directly loaded from previous nested impulse response estimate (to speed-up nested EM update)
    //this->_patlak_plot_sptr->get_impulse_response_from_parametric_image(this->impulse_response,current_estimate);
	
	//Print out the min and max values of initial impulse response (only for first and last convolution time point)
	cerr << "Nested iteration: " << nested_subiterations_num << " initial impulse response current value [conv point](min, max):" << endl;
	
	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	  {
	    if ((conv_param_num==model_array_min[1]) || (conv_param_num==model_array_max[1]))
	      {
	        const float current_min_nested_initial_impulse_response =
	          *std::min_element(this->impulse_response[conv_param_num].begin_all(),
						        this->impulse_response[conv_param_num].end_all()); 
	        const float current_max_nested_initial_impulse_response = 
              *std::max_element(this->impulse_response[conv_param_num].begin_all(),
						        this->impulse_response[conv_param_num].end_all());
	  
	        cerr << "	[" << conv_param_num << "](" << current_min_nested_initial_impulse_response << ", " << current_max_nested_initial_impulse_response << ")		" << endl;
		  }
	  }
    cerr << endl;
	
	//Then multiply the initial impulse response with the input function convolution matrix to get the dynamic images (2nd step of kinetic forward projection)
	this->_patlak_plot_sptr->get_dynamic_image_from_impulse_response(dyn_image_nested_loop_estimate,this->impulse_response);
 
    //Print out the min and max values of the forward projected frames of the dynamic image at the beginning of each nested update
    for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
 	  {
	  const float current_min_nested_dyn_image_estimate =
		  *std::min_element(dyn_image_nested_loop_estimate[frame_num].begin_all(),
						    dyn_image_nested_loop_estimate[frame_num].end_all()); 
	  const float current_max_nested_dyn_image_estimate = 
		  *std::max_element(dyn_image_nested_loop_estimate[frame_num].begin_all(),
			   			    dyn_image_nested_loop_estimate[frame_num].end_all());
	  cerr << "Nested iteration: " << nested_subiterations_num 
		     << " Forward projected dynamic image nested estimate for frame: " << frame_num << " (min, max): (" 
		     << current_min_nested_dyn_image_estimate << ", " << current_max_nested_dyn_image_estimate
		     << ") " << endl;
      }	

	//At each nested iteration, always use the dynamic image estimate from the outer loop EM update as reference
	dyn_image_estimate = dyn_image_reference_data;
	  
    //Print out the min and max values of the outer loop estimate (operating as reference) of the dynamic image at the beginning of each nested update
    for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
 	  {
	  const float current_min_nested_ref_dyn_image_estimate =
		  *std::min_element(dyn_image_estimate[frame_num].begin_all(),
						    dyn_image_estimate[frame_num].end_all()); 
	  const float current_max_nested_ref_dyn_image_estimate = 
		  *std::max_element(dyn_image_estimate[frame_num].begin_all(),
			   			    dyn_image_estimate[frame_num].end_all());
	  cerr << "Nested iteration: " << nested_subiterations_num 
		     << " Reference dynamic image estimate for frame: " << frame_num << " (min, max): (" 
		     << current_min_nested_ref_dyn_image_estimate << ", " << current_max_nested_ref_dyn_image_estimate
		     << ") " << endl;
      }	  

	
	// loop over single_frame and use model_matrix and the outer loop dynamic image estimate
	for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
	  divide(dyn_image_estimate[frame_num].begin_all(),
			 dyn_image_estimate[frame_num].end_all(),
			 dyn_image_nested_loop_estimate[frame_num].begin_all(),
			 small_num);

    //Print out the min and max values of the ratio of the reference and the forward projected dynamic frames at each nested update
    for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
 	  {			 
	  const float current_min_nested_dyn_image_ratio_estimate =
		  *std::min_element(dyn_image_estimate[frame_num].begin_all(),
						    dyn_image_estimate[frame_num].end_all()); 
	  const float current_max_nested_dyn_image_ratio_estimate = 
		  *std::max_element(dyn_image_estimate[frame_num].begin_all(),
						    dyn_image_estimate[frame_num].end_all());
	  cerr << "Nested iteration: " << nested_subiterations_num 
		   << " Dynamic images ratio for frame: " << frame_num << " (min, max): (" 
		   << current_min_nested_dyn_image_ratio_estimate << ", " << current_max_nested_dyn_image_ratio_estimate
		   << ") " << endl;
      }		   

	// Then multiply the generated dynamic image ratio factors with the transverse of the input function convolution matrix 
	// to obtain the non-normalized update factors for the impulse response (kinetic back projection)
	this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient(impulse_response_gradient,
																	    dyn_image_estimate) ;																		 
  

  	//Print out the min and max values of impulse response gradient (update factors) before sensitivity division (i.e. non-normalized), (only for first and last convolution time point)
	cerr << "Nested iteration: " << nested_subiterations_num 
	     << " sub-gradient (update factors for impulse response) before sensitivity devision current value [conv point](min, max):" << endl;
	
	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	  {	  
	    if ((conv_param_num==model_array_min[1]) || (conv_param_num==model_array_max[1]))
	      {
	        const float current_min_nested_impulse_response_gradient =
	          *std::min_element(impulse_response_gradient[conv_param_num].begin_all(),
						        impulse_response_gradient[conv_param_num].end_all()); 
	        const float current_max_nested_impulse_response_gradient = 
              *std::max_element(impulse_response_gradient[conv_param_num].begin_all(),
						        impulse_response_gradient[conv_param_num].end_all());
	  
	        cerr << "	[" << conv_param_num << "](" << current_min_nested_impulse_response_gradient << ", " << current_max_nested_impulse_response_gradient << ")		" << endl;
		  }
	  }
    cerr << endl;	  
  
    // Perform model sensitivity division for each time convolution point
  	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	  {
	  // Devide by model sensitivity
	  divide(impulse_response_gradient[conv_param_num].begin_all(), 
	         impulse_response_gradient[conv_param_num].end_all(),
		     this->sensitivity_impulse_response_image[conv_param_num].begin_all(),
		     small_num);
	  }

  	//Print out  the min and max values of impulse response gradient (update factors) after sensitivity division (only for first and last convolution time point)
	cerr << "Nested iteration: " << nested_subiterations_num 
	     << " sub-gradient (update factors for impulse response) old value: [conv point](min, max),		new value: [conv point](min, max)" << endl;
	
	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	  {
				  
	  const float new_min_nested_impulse_response = 
	    static_cast<float>(this->minimum_nested_relative_change);
	  const float new_max_nested_impulse_response = 
	    static_cast<float>(this->maximum_nested_relative_change);

	  if ((conv_param_num==model_array_min[1]) || (conv_param_num==model_array_max[1]))
	    {
		
		  const float current_min_nested_impulse_response_normalized_gradient =
	        *std::min_element(impulse_response_gradient[conv_param_num].begin_all(),
						      impulse_response_gradient[conv_param_num].end_all()); 
	      const float current_max_nested_impulse_response_normalized_gradient = 
            *std::max_element(impulse_response_gradient[conv_param_num].begin_all(),
						      impulse_response_gradient[conv_param_num].end_all());
						  
	      cerr << "	[" << conv_param_num << "](" << current_min_nested_impulse_response_normalized_gradient << ", " 
	           << current_max_nested_impulse_response_normalized_gradient << "),		"
		       << "	[" << conv_param_num << "](" << max(current_min_nested_impulse_response_normalized_gradient, new_min_nested_impulse_response) << ", " 
	           << min(current_max_nested_impulse_response_normalized_gradient, new_max_nested_impulse_response) << ")		" << endl;
	    }
		   
	  threshold_upper_lower(impulse_response_gradient[conv_param_num].begin_all(),
							impulse_response_gradient[conv_param_num].end_all(), 
							new_min_nested_impulse_response, new_max_nested_impulse_response);
	  }
	 cerr << endl;
							
							
	//Nested updates of impulse response estimates and printing of updated values
	
	cerr << "Nested iteration: " << nested_subiterations_num 
	     << " Updated impulse response value for each convolution point [conv point](min, max):" << endl;
	
	for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)
	  {
	  DiscretisedDensity<3,float>::const_full_iterator imp_response_gradient_iter = impulse_response_gradient[conv_param_num].begin_all_const(); 
	  DiscretisedDensity<3,float>::const_full_iterator end_imp_response_gradient_iter = impulse_response_gradient[conv_param_num].end_all_const(); 
	  DiscretisedDensity<3,float>::full_iterator imp_response_iter = this->impulse_response[conv_param_num].begin_all(); 
	  //Update mechanism for impulse response image
	  while (imp_response_gradient_iter!=end_imp_response_gradient_iter) 
	    { 
	    *imp_response_iter *= (*imp_response_gradient_iter); 
	    ++imp_response_iter; ++imp_response_gradient_iter; 
	    }
	
	  //Print out the min and max values of the nested updated impulse response vector for each nested iteration (first and last convolution point only)
	  if ((conv_param_num==model_array_min[1]) || (conv_param_num==model_array_max[1]))
	    {
	      const float current_min_nested_updated_impulse_response =
	        *std::min_element(this->impulse_response[conv_param_num].begin_all(),
		   			          this->impulse_response[conv_param_num].end_all()); 
	      const float current_max_nested_updated_impulse_response = 
	        *std::max_element(this->impulse_response[conv_param_num].begin_all(),
					          this->impulse_response[conv_param_num].end_all());	
	      cerr << "	[" << conv_param_num << "](" << current_min_nested_updated_impulse_response << ", " << current_max_nested_updated_impulse_response << ")		" << endl;
		}
	  }
	cerr << endl;

		 
	// Calculation of nested parametric image updates from the updated impulse response estimates
	// To speed-up nested EM updates, estimate Ki, kloss and V parametric images only for the last nested iteration
	if (nested_subiterations_num==this->num_nested_subiterations)
	  {
	  
	  cerr << "Nested EM iteration: " << nested_subiterations_num << " is the last generalized Patlak nested EM iteration... " << endl
	       << "Estimation of nested generalized Patlak parametric image estimates is now conducted... " << endl;
	
	  this->_patlak_plot_sptr->get_generalized_patlak_parameters_from_impulse_response(current_estimate, dyn_image_estimate, this->impulse_response);
	
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
   else
     {
      cerr << "Nested EM iteration: " << nested_subiterations_num << " is not the last generalized Patlak nested EM iteration (" 
          << this->num_nested_subiterations << "). Estimation of nested generalized Patlak parametric image estimates is skipped until last nested EM iteration" << endl;
     }		   

  }
  
  cerr << "End of regular nested reconstruction process of parameter estimates and impulse response (after " 
       << this->num_nested_subiterations << " nested EM subiterations)" << endl << endl;

}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
initialize_nested_loop_parameters_with_initialization_model(TargetT &parametric_gradient,
                                                            TargetT &current_estimate,
                                                            DynamicDiscretisedDensity &dyn_image_estimate,
											                DynamicDiscretisedDensity &dyn_image_reference_data,
											                DynamicDiscretisedDensity &dyn_image_nested_loop_estimate)											 
{
  
  //nested EM initialization loop
  cerr << endl << "Entering nested initialization loop (" << this->num_nested_initialization_subiterations << " subiterations)." << endl << endl;
  
  for(nested_initialization_subiterations_num=1; nested_initialization_subiterations_num<=this->num_nested_initialization_subiterations; nested_initialization_subiterations_num++)
  {
    //equivalent of forward-projection operation for kinetic parameter estimation
	this->_patlak_plot_sptr->get_dynamic_image_from_initialization_parametric_image(dyn_image_nested_loop_estimate,
	                                                                                current_estimate); 
 
    dyn_image_estimate = dyn_image_reference_data;
	// loop over single_frame and use model_matrix and the outer loop dynamic image estimate
	for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
	  divide(dyn_image_estimate[frame_num].begin_all(),
			 dyn_image_estimate[frame_num].end_all(),
			 dyn_image_nested_loop_estimate[frame_num].begin_all(),
			 small_num);

	 //equivalent of back-projection operation for kinetic parameter estimation
	this->_patlak_plot_sptr->multiply_dynamic_image_with_initialization_model_gradient(parametric_gradient,
																	                   dyn_image_estimate) ;																		 
  

	// Perform model sensitivity division and update for all nested iterations	  

	// Devide by model sensitivity
	divide(parametric_gradient.begin_all(), 
	    parametric_gradient.end_all(),
		this->initialization_model_sensitivity_image_sptr->begin_all(),
		small_num);

	if (nested_initialization_subiterations_num != 1)
	{
	  const float current_min_nested_gradient =
	    *std::min_element(parametric_gradient.begin_all(),
						  parametric_gradient.end_all()); 
	  const float current_max_nested_gradient = 
        *std::max_element(parametric_gradient.begin_all(),
						  parametric_gradient.end_all()); 
	  const float new_min_nested_gradient = 
	    static_cast<float>(this->minimum_nested_relative_change);
	  const float new_max_nested_gradient = 
	    static_cast<float>(this->maximum_nested_relative_change);
	  cerr << "Nested initialization iteration: " << nested_initialization_subiterations_num 
		   << " sub-gradient(update image) old value (min, max): (" 
		   << current_min_nested_gradient << ", " << current_max_nested_gradient
		   << "), new value (min, max) (" 
		   << max(current_min_nested_gradient, new_min_nested_gradient) << ", " 
		   << min(current_max_nested_gradient, new_max_nested_gradient) << ")" << endl;

	  threshold_upper_lower(parametric_gradient.begin_all(),
							parametric_gradient.end_all(), 
							new_min_nested_gradient, new_max_nested_gradient);      
	}

	//Nested updates of image estimates		
	{		  
	  typename TargetT::const_full_iterator parametric_gradient_iter = parametric_gradient.begin_all_const(); 
	  const typename TargetT::const_full_iterator end_parametric_gradient_iter = parametric_gradient.end_all_const(); 
	  typename TargetT::full_iterator current_estimate_iter = current_estimate.begin_all(); 
	  while (parametric_gradient_iter!=end_parametric_gradient_iter) 
	  { 
		*current_estimate_iter *= (*parametric_gradient_iter); 
		++current_estimate_iter; ++parametric_gradient_iter;  
	  } 
	}
	  
	//Print out the min and max values of the nested updated image for each nested iteration
	const float current_min_nested_updated_image =
	*std::min_element(current_estimate.begin_all(),
					  current_estimate.end_all()); 
	const float current_max_nested_updated_image = 
	*std::max_element(current_estimate.begin_all(),
					  current_estimate.end_all());
	cerr << "Nested initialization iteration: " << nested_initialization_subiterations_num 
	     << " Updated image value (min, max) ("
		 << current_min_nested_updated_image << ", " << current_max_nested_updated_image << ")" << endl << endl;
	
  }
  
  cerr << "End of initialization process of parameter estimates and impulse response (after " 
       << this->num_nested_initialization_subiterations << " nested initialization EM subiterations)" << endl << endl;
}


template<typename TargetT>
double
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                  const int subset_num)
{
  assert(subset_num>=0);
  assert(subset_num<this->num_subsets);

  double result = 0.;
  DynamicDiscretisedDensity dyn_image_estimate=this->_dyn_image_template;

  // TODO why fill with 1?
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
    std::fill(dyn_image_estimate[frame_num].begin_all(),
              dyn_image_estimate[frame_num].end_all(),
              1.F);
  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_image_estimate,current_estimate) ; 
 
  // loop over single_frame
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();
      frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();
      ++frame_num)
    {
      result +=
        this->_single_frame_obj_funcs[frame_num].
        compute_objective_function_without_penalty(dyn_image_estimate[frame_num], 
                                                   subset_num);
    }
  return result;
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
compute_model_sensitivity_impulse_response(DynamicDiscretisedDensity& impulse_response_image)
{

  //Initialize model sensitivity image	
  //for(int conv_param_num = model_array_min[1];conv_param_num<=model_array_max[1] ; ++conv_param_num)  
  //  std::fill(impulse_response_image[conv_param_num].begin_all(),
  //            impulse_response_image[conv_param_num].end_all(),
  //	          1.F);
  
  DynamicDiscretisedDensity dyn_image_of_all_ones=this->_dyn_image_template;

  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
      std::fill(dyn_image_of_all_ones[frame_num].begin_all(),
	    dyn_image_of_all_ones[frame_num].end_all(),
	    1.F);
		
  cerr << "Computing impulse response sensitivity image..." << endl;

  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient(impulse_response_image,
																	  dyn_image_of_all_ones);
																	
																						
  //Print out the min and max values of the model sensitivity image
  const float current_min_model_sensitivity =
  *std::min_element(impulse_response_image.begin_all(),
   				    impulse_response_image.end_all()); 
  const float current_max_model_sensitivity = 
  *std::max_element(impulse_response_image.begin_all(),
				    impulse_response_image.end_all());
  cerr << "Impulse response sensitivity image " 
	   << ", (min, max): (" << current_min_model_sensitivity << ", " << current_max_model_sensitivity << ")" << endl;
  
  cerr << "Impulse_response sensitivity image has been computed." << endl; 
}  


template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
compute_initialization_model_sensitivity_image(TargetT& param_image)
{

  //Initialize the initialization model sensitivity image
  shared_ptr<TargetT> param_image_sptr(param_image.get_empty_copy());
  this->initialization_model_sensitivity_image_sptr=param_image_sptr;
  
  std::fill(initialization_model_sensitivity_image_sptr->begin_all(),
      this->initialization_model_sensitivity_image_sptr->end_all(),
  	    1.F);
  
  DynamicDiscretisedDensity dyn_image_of_all_ones=this->_dyn_image_template;

  // loop over single_frame and use model_matrix
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
      std::fill(dyn_image_of_all_ones[frame_num].begin_all(),
	    dyn_image_of_all_ones[frame_num].end_all(),
	    1.F);
		
  cerr << "Computing initialization model sensitivity image..." << endl;

  this->_patlak_plot_sptr->multiply_dynamic_image_with_initialization_model_gradient(*this->initialization_model_sensitivity_image_sptr,
																	                 dyn_image_of_all_ones);
																	
																						
  //Print out the min and max values of the initialization model sensitivity image
  const float current_min_initialization_model_sensitivity =
  *std::min_element(this->initialization_model_sensitivity_image_sptr->begin_all(),
   				    this->initialization_model_sensitivity_image_sptr->end_all()); 
  const float current_max_initialization_model_sensitivity = 
  *std::max_element(this->initialization_model_sensitivity_image_sptr->begin_all(),
				    this->initialization_model_sensitivity_image_sptr->end_all());
  cerr << "Initialization Model sensitivity image " 
	   << ", (min, max): (" << current_min_initialization_model_sensitivity << ", " << current_max_initialization_model_sensitivity << ")" << endl;
  
  cerr << "Initialization Model sensitivity image has been computed." << endl; 
}  



template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
{
  DynamicDiscretisedDensity dyn_image_of_all_ones=this->_dyn_image_template;
  DynamicDiscretisedDensity sensitivity_impulse_response=this->_imp_response_image_template;
  
  // loop over single_frame and use model_matrix
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
      std::fill(dyn_image_of_all_ones[frame_num].begin_all(),
	    dyn_image_of_all_ones[frame_num].end_all(),
	    1.F);

  this->add_subset_impulse_response_sensitivity(sensitivity_impulse_response,subset_num);
  
  this->_patlak_plot_sptr->get_generalized_patlak_parameters_from_impulse_response(sensitivity, dyn_image_of_all_ones, sensitivity_impulse_response);
																						
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
add_subset_initialization_sensitivity(TargetT& initialization_model_sensitivity, const int subset_num) const
{
  DynamicDiscretisedDensity dyn_image_of_all_ones=this->_dyn_image_template;

  // loop over single_frame and use model_matrix
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
      std::fill(dyn_image_of_all_ones[frame_num].begin_all(),
	    dyn_image_of_all_ones[frame_num].end_all(),
	    1.F);

  this->_patlak_plot_sptr->multiply_dynamic_image_with_initialization_model_gradient_and_add_to_input(initialization_model_sensitivity,
																						                      dyn_image_of_all_ones);
																						
 
}

template<typename TargetT>
void
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
add_subset_impulse_response_sensitivity(DynamicDiscretisedDensity& impulse_response, const int subset_num) const
{
  DynamicDiscretisedDensity dyn_image_of_all_ones=this->_dyn_image_template;

  // loop over single_frame and use model_matrix
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();++frame_num)
      std::fill(dyn_image_of_all_ones[frame_num].begin_all(),
	    dyn_image_of_all_ones[frame_num].end_all(),
	    1.F);

		
  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient_and_add_to_input(impulse_response,
																						dyn_image_of_all_ones);
																						
 
}

template<typename TargetT>
Succeeded
PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>::
actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                       const TargetT& input,
                                                                       const int subset_num) const
{
  {
    string explanation;
    if (!input.has_same_characteristics(this->get_sensitivity(), 
                                        explanation))
      {
        warning("PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData:\n"
                "sensitivity and input for add_multiplication_with_approximate_Hessian_without_penalty\n"
                "should have the same characteristics.\n%s",
                explanation.c_str());
        return Succeeded::no;
      }
  }   
#ifndef NDEBUG
  std::cerr << "INPUT max: (" << input.construct_single_density(1).find_max()
            << " , " << input.construct_single_density(2).find_max()
            << ")\n";
#endif //NDEBUG
  DynamicDiscretisedDensity dyn_input=this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_output=this->_dyn_image_template;
  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_input,input) ; 

  VectorWithOffset<float> scale_factor(this->_patlak_plot_sptr->get_starting_frame(),this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames());
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();
      frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();
      ++frame_num)
    {
      assert(dyn_input[frame_num].find_max()==dyn_input[frame_num].find_min());
      if (dyn_input[frame_num].find_max()==dyn_input[frame_num].find_min() && dyn_input[frame_num].find_min()>0.F)
        scale_factor[frame_num]=dyn_input[frame_num].find_max();
      else
        error("The input image should be uniform even after multiplying with the Patlak Plot.\n");

/*! /note This is used to avoid higher values than these set in the precompute_denominator_of_conditioner_without_penalty() function. 
/sa for more information see the recon_array_functions.cxx and the value of the max_quotient (originaly set to 10000.F)
*/
      dyn_input[frame_num]/=scale_factor[frame_num]; 
#ifndef NDEBUG
      std::cerr << "scale factor[" << frame_num << "] " << scale_factor[frame_num] << "\n";
      std::cerr << "dyn_input[" << frame_num << "] max after scale: " 
                << dyn_input[frame_num].find_max() << "\n";
#endif //NDEBUG
      this->_single_frame_obj_funcs[frame_num].
        add_multiplication_with_approximate_sub_Hessian_without_penalty(dyn_output[frame_num],
                                                                        dyn_input[frame_num],
                                                                        subset_num);      
#ifndef NDEBUG
      std::cerr << "dyn_output[" << frame_num << "] max before scale: (" 
                << dyn_output[frame_num].find_max() << "\n";
#endif //NDEBUG
      dyn_output[frame_num]*=scale_factor[frame_num];
#ifndef NDEBUG
      std::cerr << "dyn_output[" << frame_num << "] max after scale: (" 
                << dyn_output[frame_num].find_max() << "\n";
#endif //NDEBUG
    } // end of loop over frames
  shared_ptr<TargetT> unnormalised_temp(output.get_empty_copy());
  DynamicDiscretisedDensity unnormalised_imp_response;
  shared_ptr<TargetT> temp(output.get_empty_copy());
  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient(unnormalised_imp_response,
                                                                      dyn_output) ;
  
  this->_patlak_plot_sptr->get_generalized_patlak_parameters_from_impulse_response(*unnormalised_temp,dyn_output,unnormalised_imp_response);
  // Trick to use a better step size for the two parameters. 
  (this->_patlak_plot_sptr->get_model_matrix()).normalise_parametric_image_with_model_sum(*temp,*unnormalised_temp,this->_patlak_plot_sptr->_num_conv_params) ;
#ifndef NDEBUG
  std::cerr << "TEMP max: (" << temp->construct_single_density(1).find_max()
            << " , " << temp->construct_single_density(2).find_max()
            << ")\n";
  // Writing images
  OutputFileFormat<GeneralizedPatlakVoxelsOnCartesianGrid>::default_sptr()->write_to_file("all_params_one_input.img", input);
  OutputFileFormat<GeneralizedPatlakVoxelsOnCartesianGrid>::default_sptr()->write_to_file("temp_denominator.img", *temp);
  dyn_input.write_to_ecat7("dynamic_input_from_all_params_one.img");
  dyn_output.write_to_ecat7("dynamic_precomputed_denominator.img");
  DynamicProjData temp_projdata = this->get_dyn_proj_data();
  for(unsigned int frame_num=this->_patlak_plot_sptr->get_starting_frame();
      frame_num<=this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames();
      ++frame_num)
    temp_projdata.set_proj_data_sptr(this->_single_frame_obj_funcs[frame_num].get_proj_data_sptr(),frame_num);
    
  temp_projdata.write_to_ecat7("DynamicProjections.S");
#endif // NDEBUG
  // output += temp
  typename TargetT::full_iterator out_iter = output.begin_all();
  typename TargetT::full_iterator out_end = output.end_all();
  typename TargetT::const_full_iterator temp_iter = temp->begin_all_const();
  while (out_iter != out_end)
    {
      *out_iter += *temp_iter;
      ++out_iter; ++temp_iter;
    }
#ifndef NDEBUG
  std::cerr << "OUTPUT max: (" << output.construct_single_density(1).find_max()
            << " , " << output.construct_single_density(2).find_max()
            << ")\n";
#endif // NDEBUG

  
  return Succeeded::yes;
}


END_NAMESPACE_STIR

