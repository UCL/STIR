//
// PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.txx
//
/*
  Copyright (C) 2006 - 2011-01-14 Hammersmith Imanet Ltd
  Copyright (C) 2011 Kris Thielemans
  Copyright (C) 2013m 2018, University College London

  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \ingroup modelling
  \brief Implementation of class stir::PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Charalampos Tsoumpas
*/
#include "stir/DiscretisedDensity.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

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
#include "stir/recon_buildblock/BinNormalisationWithCalibration.h"

#include <algorithm>
#include <string>
// For the Patlak Plot Modelling
#include "stir/modelling/ModelMatrix.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.h"
#ifndef NDEBUG
#  include "stir/IO/write_to_file.h"
#endif

START_NAMESPACE_STIR

template <typename TargetT>
const char* const PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::registered_name
    = "PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData";

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::set_defaults()
{
  base_type::set_defaults();

  this->_input_filename = "";
  this->_max_segment_num_to_process = -1; // use all segments
  // num_views_to_add=1;    // KT 20/06/2001 disabled

  this->_dyn_proj_data_sptr.reset();
  this->_zero_seg0_end_planes = 0;

  this->_additive_dyn_proj_data_filename = "0";
  this->_additive_dyn_proj_data_sptr.reset();

#ifndef USE_PMRT // set default for _projector_pair_ptr
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingRayTracing());
  shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingInterpolation());
#else
  shared_ptr<ProjMatrixByBin> PM(new ProjMatrixByBinUsingRayTracing());
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
  shared_ptr<BackProjectorByBin> back_projector_ptr(new BackProjectorByBinUsingProjMatrixByBin(PM));
#endif

  this->_projector_pair_ptr.reset(new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));
  this->_normalisation_sptr.reset(new TrivialBinNormalisation);

  this->target_parameter_parser.set_defaults();

  // Modelling Stuff
  this->_patlak_plot_sptr.reset();
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData Parameters");
  this->parser.add_stop_key("End PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData Parameters");
  this->parser.add_key("input file", &this->_input_filename);

  // parser.add_key("mash x views", &num_views_to_add);   // KT 20/06/2001 disabled
  this->parser.add_key("maximum absolute segment number to process", &this->_max_segment_num_to_process);
  this->parser.add_key("zero end planes of segment 0", &this->_zero_seg0_end_planes);

  this->target_parameter_parser.add_to_keymap(this->parser);
  this->parser.add_parsing_key("Projector pair type", &this->_projector_pair_ptr);

  // Scatter correction
  this->parser.add_key("additive sinograms", &this->_additive_dyn_proj_data_filename);

  // normalisation (and attenuation correction)
  this->parser.add_parsing_key("Bin Normalisation type", &this->_normalisation_sptr);

  // Modelling Information
  this->parser.add_parsing_key("Kinetic Model Type", &this->_patlak_plot_sptr); // Do sth with dynamic_cast to get the PatlakPlot

  // Regularization Information
  //  this->parser.add_parsing_key("prior type", &this->_prior_sptr);
}

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  if (this->_input_filename.length() == 0)
    {
      warning("You need to specify an input filename");
      return true;
    }

#if 0 // KT 20/06/2001 disabled as not functional yet
  if (num_views_to_add!=1 && (num_views_to_add<=0 || num_views_to_add%2 != 0))
    { warning("The 'mash x views' key has an invalid value (must be 1 or even number)"); return true; }
#endif

  this->_dyn_proj_data_sptr = DynamicProjData::read_from_file(_input_filename);
  if (is_null_ptr(this->_dyn_proj_data_sptr))
    {
      warning("Error reading input file %s", _input_filename.c_str());
      return true;
    }
  this->target_parameter_parser.check_values();

  if (this->_additive_dyn_proj_data_filename != "0")
    {
      info(format("Reading additive projdata data {}", this->_additive_dyn_proj_data_filename));
      this->_additive_dyn_proj_data_sptr = DynamicProjData::read_from_file(this->_additive_dyn_proj_data_filename);
      if (is_null_ptr(this->_additive_dyn_proj_data_sptr))
        {
          warning("Error reading additive input file %s", _additive_dyn_proj_data_filename.c_str());
          return true;
        }
    }
  return false;
}

template <typename TargetT>
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<
    TargetT>::PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData()
{
  this->set_defaults();
}

template <typename TargetT>
TargetT*
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::construct_target_ptr() const
{
  return this->target_parameter_parser.create(this->get_input_data());
}

template <typename TargetT>
std::unique_ptr<ExamInfo>
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_exam_info_uptr_for_target() const
{
  auto exam_info_uptr = this->get_exam_info_uptr_for_target();
  if (auto norm_ptr = dynamic_cast<BinNormalisationWithCalibration const* const>(get_normalisation_sptr().get()))
    {
      exam_info_uptr->set_calibration_factor(norm_ptr->get_calibration_factor());
      // somehow tell the image that it's calibrated (do we have a way?)
    }
  else
    {
      exam_info_uptr->set_calibration_factor(1.F);
      // somehow tell the image that it's not calibrated (do we have a way?)
    }
  return exam_info_uptr;
}

/***************************************************************
  subset balancing
***************************************************************/

template <typename TargetT>
bool
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::actual_subsets_are_approximately_balanced(
    std::string& warning_message) const
{ // call actual_subsets_are_approximately_balanced( for first single_frame_obj_func
  if (this->_patlak_plot_sptr->get_time_frame_definitions().get_num_frames() == 0 || this->_single_frame_obj_funcs.size() == 0)
    error("PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData:\n"
          "actual_subsets_are_approximately_balanced called but not frames yet.\n");
  else if (this->_single_frame_obj_funcs.size() != 0)
    {
      bool frames_are_balanced = true;
      for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
           frame_num <= this->_patlak_plot_sptr->get_ending_frame();
           ++frame_num)
        frames_are_balanced &= this->_single_frame_obj_funcs[frame_num].subsets_are_approximately_balanced(warning_message);
      return frames_are_balanced;
    }
  else
    error("Something strange happened in PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData:\n"
          "actual_subsets_are_approximately_balanced called before setup()?\n");
  return false;
}

/***************************************************************
  get_ functions
***************************************************************/
template <typename TargetT>
const DynamicProjData&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_dyn_proj_data() const
{
  return *this->_dyn_proj_data_sptr;
}

template <typename TargetT>
const shared_ptr<DynamicProjData>&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_dyn_proj_data_sptr() const
{
  return this->_dyn_proj_data_sptr;
}

template <typename TargetT>
const int
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_max_segment_num_to_process() const
{
  return this->_max_segment_num_to_process;
}

template <typename TargetT>
const bool
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_zero_seg0_end_planes() const
{
  return this->_zero_seg0_end_planes;
}

template <typename TargetT>
const DynamicProjData&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_additive_dyn_proj_data() const
{
  return *this->_additive_dyn_proj_data_sptr;
}

template <typename TargetT>
const shared_ptr<DynamicProjData>&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_additive_dyn_proj_data_sptr() const
{
  return this->_additive_dyn_proj_data_sptr;
}

template <typename TargetT>
const ProjectorByBinPair&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_projector_pair() const
{
  return *this->_projector_pair_ptr;
}

template <typename TargetT>
const shared_ptr<ProjectorByBinPair>&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_projector_pair_sptr() const
{
  return this->_projector_pair_ptr;
}

template <typename TargetT>
const BinNormalisation&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_normalisation() const
{
  return *this->_normalisation_sptr;
}

template <typename TargetT>
const shared_ptr<BinNormalisation>&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_normalisation_sptr() const
{
  return this->_normalisation_sptr;
}

/***************************************************************
  set_ functions
***************************************************************/
template <typename TargetT>
int
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::set_num_subsets(const int num_subsets)
{
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    {
      if (this->_single_frame_obj_funcs.size() != 0)
        if (this->_single_frame_obj_funcs[frame_num].set_num_subsets(num_subsets) != num_subsets)
          error("set_num_subsets didn't work");
    }
  this->num_subsets = num_subsets;
  return this->num_subsets;
}

/***************************************************************
  set_up()
***************************************************************/
template <typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::set_up_before_sensitivity(
    shared_ptr<const TargetT> const& target_sptr)
{
  if (this->_max_segment_num_to_process == -1)
    this->_max_segment_num_to_process = (this->_dyn_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num();

  if (this->_max_segment_num_to_process > (this->_dyn_proj_data_sptr)->get_proj_data_sptr(1)->get_max_segment_num())
    {
      warning("_max_segment_num_to_process (%d) is too large", this->_max_segment_num_to_process);
      return Succeeded::no;
    }

  const shared_ptr<ProjDataInfo> proj_data_info_sptr(
      (this->_dyn_proj_data_sptr->get_proj_data_sptr(1))->get_proj_data_info_sptr()->clone());
  proj_data_info_sptr->reduce_segment_range(-this->_max_segment_num_to_process, +this->_max_segment_num_to_process);

  if (is_null_ptr(this->_projector_pair_ptr))
    {
      warning("You need to specify a projector pair");
      return Succeeded::no;
    }

  if (this->num_subsets <= 0)
    {
      warning("Number of subsets %d should be larger than 0.", this->num_subsets);
      return Succeeded::no;
    }

  if (is_null_ptr(this->_normalisation_sptr))
    {
      warning("Invalid normalisation object");
      return Succeeded::no;
    }

  if (this->_normalisation_sptr->set_up(this->_dyn_proj_data_sptr->get_exam_info_sptr(), proj_data_info_sptr) == Succeeded::no)
    return Succeeded::no;

  if (this->_patlak_plot_sptr->set_up() == Succeeded::no)
    return Succeeded::no;

  if (this->_patlak_plot_sptr->get_starting_frame() <= 0
      || this->_patlak_plot_sptr->get_starting_frame() > this->_patlak_plot_sptr->get_ending_frame())
    {
      warning("Starting frame is %d. Generally, it should be a late frame,\nbut in any case it should be less than the number of "
              "frames %d\nand at least 1.",
              this->_patlak_plot_sptr->get_starting_frame(),
              this->_patlak_plot_sptr->get_ending_frame());
      return Succeeded::no;
    }
  {
    const shared_ptr<DiscretisedDensity<3, float>> density_template_sptr(
        (target_sptr->construct_single_density(1)).get_empty_copy());
    const shared_ptr<Scanner> scanner_sptr(new Scanner(*proj_data_info_sptr->get_scanner_ptr()));
    this->_dyn_image_template = DynamicDiscretisedDensity(this->_patlak_plot_sptr->get_time_frame_definitions(),
                                                          this->_dyn_proj_data_sptr->get_start_time_in_secs_since_1970(),
                                                          scanner_sptr,
                                                          density_template_sptr);

    // construct _single_frame_obj_funcs
    this->_single_frame_obj_funcs.resize(this->_patlak_plot_sptr->get_starting_frame(),
                                         this->_patlak_plot_sptr->get_ending_frame());

    for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
         frame_num <= this->_patlak_plot_sptr->get_ending_frame();
         ++frame_num)
      {
        this->_single_frame_obj_funcs[frame_num].set_projector_pair_sptr(this->_projector_pair_ptr);
        this->_single_frame_obj_funcs[frame_num].set_proj_data_sptr(this->_dyn_proj_data_sptr->get_proj_data_sptr(frame_num));
        this->_single_frame_obj_funcs[frame_num].set_max_segment_num_to_process(this->_max_segment_num_to_process);
        this->_single_frame_obj_funcs[frame_num].set_zero_seg0_end_planes(this->_zero_seg0_end_planes != 0);
        if (!is_null_ptr(this->_additive_dyn_proj_data_sptr))
          this->_single_frame_obj_funcs[frame_num].set_additive_proj_data_sptr(
              this->_additive_dyn_proj_data_sptr->get_proj_data_sptr(frame_num));
        this->_single_frame_obj_funcs[frame_num].set_num_subsets(this->num_subsets);
        this->_single_frame_obj_funcs[frame_num].set_frame_num(frame_num);
        this->_single_frame_obj_funcs[frame_num].set_frame_definitions(this->_patlak_plot_sptr->get_time_frame_definitions());
        this->_single_frame_obj_funcs[frame_num].set_normalisation_sptr(this->_normalisation_sptr);
        this->_single_frame_obj_funcs[frame_num].set_recompute_sensitivity(this->get_recompute_sensitivity());
        this->_single_frame_obj_funcs[frame_num].set_use_subset_sensitivities(this->get_use_subset_sensitivities());
        if (this->_single_frame_obj_funcs[frame_num].set_up(density_template_sptr) != Succeeded::yes)
          error("Single frame objective functions is not set correctly!");
      }
  } //_single_frame_obj_funcs[frame_num]

  return Succeeded::yes;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::set_input_data(const shared_ptr<ExamData>& arg)
{
  this->_dyn_proj_data_sptr = dynamic_pointer_cast<DynamicProjData>(arg);
}

template <typename TargetT>
const DynamicProjData&
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::get_input_data() const
{
  return *this->_dyn_proj_data_sptr;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::set_additive_proj_data_sptr(
    const shared_ptr<ExamData>& arg)
{
  this->_additive_dyn_proj_data_sptr = dynamic_pointer_cast<DynamicProjData>(arg);
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::set_normalisation_sptr(
    const shared_ptr<BinNormalisation>& arg)
{
  //  this->normalisation_sptr = arg;
  error("Not implemeted yet");
}

/*************************************************************************
  functions that compute the value/gradient of the objective function etc
*************************************************************************/

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::actual_compute_subset_gradient_without_penalty(
    TargetT& gradient, const TargetT& current_estimate, const int subset_num, const bool add_sensitivity)
{
  if (subset_num < 0 || subset_num >= this->get_num_subsets())
    error("compute_sub_gradient_without_penalty subset_num out-of-range error");

  DynamicDiscretisedDensity dyn_gradient = this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_image_estimate = this->_dyn_image_template;

  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    std::fill(dyn_image_estimate[frame_num].begin_all(), dyn_image_estimate[frame_num].end_all(), 1.F);

  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_image_estimate, current_estimate);

  // loop over single_frame and use model_matrix
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    {
      std::fill(dyn_gradient[frame_num].begin_all(), dyn_gradient[frame_num].end_all(), 1.F);

      this->_single_frame_obj_funcs[frame_num].actual_compute_subset_gradient_without_penalty(
          dyn_gradient[frame_num], dyn_image_estimate[frame_num], subset_num, add_sensitivity);
    }

  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient(gradient, dyn_gradient);
}

template <typename TargetT>
double
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::actual_compute_objective_function_without_penalty(
    const TargetT& current_estimate, const int subset_num)
{
  assert(subset_num >= 0);
  assert(subset_num < this->num_subsets);

  double result = 0.;
  DynamicDiscretisedDensity dyn_image_estimate = this->_dyn_image_template;

  // TODO why fill with 1?
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    std::fill(dyn_image_estimate[frame_num].begin_all(), dyn_image_estimate[frame_num].end_all(), 1.F);
  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_image_estimate, current_estimate);

  // loop over single_frame
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    {
      result += this->_single_frame_obj_funcs[frame_num].compute_objective_function_without_penalty(dyn_image_estimate[frame_num],
                                                                                                    subset_num);
    }
  return result;
}

template <typename TargetT>
void
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>::add_subset_sensitivity(TargetT& sensitivity,
                                                                                                    const int subset_num) const
{
  DynamicDiscretisedDensity dyn_sensitivity = this->_dyn_image_template;

  // loop over single_frame and use model_matrix
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    {
      dyn_sensitivity[frame_num] = this->_single_frame_obj_funcs[frame_num].get_subset_sensitivity(subset_num);
      //  add_subset_sensitivity(dyn_sensitivity[frame_num],subset_num);
    }

  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient_and_add_to_input(sensitivity, dyn_sensitivity);
}

template <typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<
    TargetT>::actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                                     const TargetT& input,
                                                                                     const int subset_num) const
{
  {
    std::string explanation;
    if (!input.has_same_characteristics(this->get_sensitivity(), explanation))
      {
        warning("PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData:\n"
                "sensitivity and input for add_multiplication_with_approximate_Hessian_without_penalty\n"
                "should have the same characteristics.\n%s",
                explanation.c_str());
        return Succeeded::no;
      }
  }
#ifndef NDEBUG
  info(
      format("INPUT max: ({} , {})", input.construct_single_density(1).find_max(), input.construct_single_density(2).find_max()));
#endif // NDEBUG
  DynamicDiscretisedDensity dyn_input = this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_output = this->_dyn_image_template;
  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_input, input);

  VectorWithOffset<float> scale_factor(this->_patlak_plot_sptr->get_starting_frame(),
                                       this->_patlak_plot_sptr->get_ending_frame());
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    {
      assert(dyn_input[frame_num].find_max() == dyn_input[frame_num].find_min());
      if (dyn_input[frame_num].find_max() == dyn_input[frame_num].find_min() && dyn_input[frame_num].find_min() > 0.F)
        scale_factor[frame_num] = dyn_input[frame_num].find_max();
      else
        error("The input image should be uniform even after multiplying with the Patlak Plot.\n");

      /*! /note This is used to avoid higher values than these set in the precompute_denominator_of_conditioner_without_penalty()
      function. /sa for more information see the recon_array_functions.cxx and the value of the max_quotient (originaly set to
      10000.F)
      */
      dyn_input[frame_num] /= scale_factor[frame_num];
#ifndef NDEBUG
      info(format("scale factor[{}]: {}", frame_num, scale_factor[frame_num]));
      info(format("dyn_input[{}] max after scale: {}", frame_num, dyn_input[frame_num].find_max()));
#endif // NDEBUG
      this->_single_frame_obj_funcs[frame_num].add_multiplication_with_approximate_sub_Hessian_without_penalty(
          dyn_output[frame_num], dyn_input[frame_num], subset_num);
#ifndef NDEBUG
      info(format("dyn_output[{}] max before scale: {}", frame_num, dyn_output[frame_num].find_max()));
#endif // NDEBUG
      dyn_output[frame_num] *= scale_factor[frame_num];
#ifndef NDEBUG
      info(format("dyn_output[{}] max after scale: {}", frame_num, dyn_output[frame_num].find_max()));
#endif // NDEBUG
    }  // end of loop over frames
  shared_ptr<TargetT> unnormalised_temp(output.get_empty_copy());
  shared_ptr<TargetT> temp(output.get_empty_copy());
  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient(*unnormalised_temp, dyn_output);
  // Trick to use a better step size for the two parameters.
  (this->_patlak_plot_sptr->get_model_matrix()).normalise_parametric_image_with_model_sum(*temp, *unnormalised_temp);
#ifndef NDEBUG
  info(format("TEMP max: ({} , {})", temp->construct_single_density(1).find_max(), temp->construct_single_density(2).find_max()));
  // Writing images
  write_to_file("all_params_one_input.img", input);
  write_to_file("temp_denominator.img", *temp);
  dyn_input.write_to_ecat7("dynamic_input_from_all_params_one.img");
  dyn_output.write_to_ecat7("dynamic_precomputed_denominator.img");
  DynamicProjData temp_projdata = this->get_dyn_proj_data();
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    temp_projdata.set_proj_data_sptr(this->_single_frame_obj_funcs[frame_num].get_proj_data_sptr(), frame_num);

  temp_projdata.write_to_ecat7("DynamicProjections.S");
#endif // NDEBUG
  // output += temp
  typename TargetT::full_iterator out_iter = output.begin_all();
  typename TargetT::full_iterator out_end = output.end_all();
  typename TargetT::const_full_iterator temp_iter = temp->begin_all_const();
  while (out_iter != out_end)
    {
      *out_iter += *temp_iter;
      ++out_iter;
      ++temp_iter;
    }
#ifndef NDEBUG
  info(format(
      "OUTPUT max: ({} , {})", output.construct_single_density(1).find_max(), output.construct_single_density(2).find_max()));
#endif // NDEBUG

  return Succeeded::yes;
}

template <typename TargetT>
Succeeded
PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<
    TargetT>::actual_accumulate_sub_Hessian_times_input_without_penalty(TargetT& output,
                                                                        const TargetT& current_image_estimate,
                                                                        const TargetT& input,
                                                                        const int subset_num) const
{
  { // check argument characteristics
    std::string explanation;
    if (!input.has_same_characteristics(this->get_sensitivity(), explanation))
      {
        warning("PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData:\n"
                "sensitivity and input for actual_accumulate_sub_Hessian_times_input_without_penalty\n"
                "should have the same characteristics.\n%s",
                explanation.c_str());
        return Succeeded::no;
      }

    if (!current_image_estimate.has_same_characteristics(this->get_sensitivity(), explanation))
      {
        warning("PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData:\n"
                "sensitivity and current_image_estimate for actual_accumulate_sub_Hessian_times_input_without_penalty\n"
                "should have the same characteristics.\n%s",
                explanation.c_str());
        return Succeeded::no;
      }
  }
#ifndef NDEBUG
  info(
      format("INPUT max: ({} , {})", input.construct_single_density(1).find_max(), input.construct_single_density(2).find_max()));
#endif // NDEBUG
  DynamicDiscretisedDensity dyn_input = this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_current_image_estimate = this->_dyn_image_template;
  DynamicDiscretisedDensity dyn_output = this->_dyn_image_template;
  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_input, input);
  this->_patlak_plot_sptr->get_dynamic_image_from_parametric_image(dyn_current_image_estimate, current_image_estimate);

  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    {
      assert(dyn_input[frame_num].find_max() == dyn_input[frame_num].find_min());
      this->_single_frame_obj_funcs[frame_num].accumulate_sub_Hessian_times_input_without_penalty(
          dyn_output[frame_num], dyn_current_image_estimate[frame_num], dyn_input[frame_num], subset_num);
    } // end of loop over frames
  shared_ptr<TargetT> unnormalised_temp(output.get_empty_copy());
  shared_ptr<TargetT> temp(output.get_empty_copy());
  this->_patlak_plot_sptr->multiply_dynamic_image_with_model_gradient(*unnormalised_temp, dyn_output);
  // Trick to use a better step size for the two parameters.
  (this->_patlak_plot_sptr->get_model_matrix()).normalise_parametric_image_with_model_sum(*temp, *unnormalised_temp);
#ifndef NDEBUG
  info(format("TEMP max: ({} , {})", temp->construct_single_density(1).find_max(), temp->construct_single_density(2).find_max()));
  // Writing images
  write_to_file("all_params_one_input.img", input);
  write_to_file("temp_denominator.img", *temp);
  dyn_input.write_to_ecat7("dynamic_input_from_all_params_one.img");
  dyn_output.write_to_ecat7("dynamic_precomputed_denominator.img");
  DynamicProjData temp_projdata = this->get_dyn_proj_data();
  for (unsigned int frame_num = this->_patlak_plot_sptr->get_starting_frame();
       frame_num <= this->_patlak_plot_sptr->get_ending_frame();
       ++frame_num)
    temp_projdata.set_proj_data_sptr(this->_single_frame_obj_funcs[frame_num].get_proj_data_sptr(), frame_num);

  temp_projdata.write_to_ecat7("DynamicProjections.S");
#endif // NDEBUG
  // output += temp
  typename TargetT::full_iterator out_iter = output.begin_all();
  typename TargetT::full_iterator out_end = output.end_all();
  typename TargetT::const_full_iterator temp_iter = temp->begin_all_const();
  while (out_iter != out_end)
    {
      *out_iter += *temp_iter;
      ++out_iter;
      ++temp_iter;
    }
#ifndef NDEBUG
  info(format(
      "OUTPUT max: ({} , {})", output.construct_single_density(1).find_max(), output.construct_single_density(2).find_max()));
#endif // NDEBUG

  return Succeeded::yes;
}

END_NAMESPACE_STIR
