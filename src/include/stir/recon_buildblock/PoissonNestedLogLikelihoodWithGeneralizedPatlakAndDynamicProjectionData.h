//
/*
    Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \ingroup modelling
  \brief Declaration of class stir::PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData

  \author Nicolas A Karakatsanis

*/

#ifndef __stir_recon_buildblock_PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData_H__
#define __stir_recon_buildblock_PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData_H__
#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/Array.h"
#include "stir/BasicCoordinate.h"
#include "stir/VectorWithOffset.h"
#include "stir/DynamicProjData.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParseAndCreateParametricDiscretisedDensityFrom.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/modelling/GeneralizedPatlakPlot.h"

START_NAMESPACE_STIR

/*!
  \ingroup GeneralisedObjectiveFunction
  \ingroup modelling
  \brief a base class for LogLikelihood of independent Poisson variables
  where the mean values are non-linear combinations of the kinetic parameters.

  \par Parameters for parsing

*/

template <typename TargetT>
class PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData
    : public RegisteredParsingObject<PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>,
                                     GeneralisedObjectiveFunction<TargetT>,
                                     PoissonLogLikelihoodWithLinearModelForMean<TargetT>>
{
private:
  typedef RegisteredParsingObject<PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData<TargetT>,
                                  GeneralisedObjectiveFunction<TargetT>,
                                  PoissonLogLikelihoodWithLinearModelForMean<TargetT>>
      base_type;
  typedef PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float>> SingleFrameObjFunc;
  VectorWithOffset<SingleFrameObjFunc> _single_frame_obj_funcs;

public:
  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char* const registered_name;

  PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData();

  //! Returns a pointer to a newly allocated target object (with 0 data).
  /*! Dimensions etc are set from the \a dyn_proj_data_sptr and other information set by parsing,
    such as \c zoom, \c output_image_size_z etc.
  */
  TargetT* construct_target_ptr() const override;

  // Computes the outer loop gradient after conducting nested iterations
  /* At each nested iteration \current_estimate  is updated and therefore it
     is declared as TargetT and NOT as const TargetT
  */

  void actual_compute_subset_gradient_without_penalty(TargetT& gradient,
                                                      const TargetT& current_estimate,
                                                      const int subset_num,
                                                      const bool add_sensitivity) override;

protected:
  virtual void actual_compute_nested_sub_gradient_without_penalty(TargetT& gradient,
                                                                  TargetT& current_estimate,
                                                                  const int subset_num,
                                                                  const bool add_sensitivity);

  // The nested EM reconstruction method using the standard Patlak model for initialization purposes
  // Use this method in GeneralizedPatlak objective function class ONLY for EM initialization
  virtual void
  initialize_nested_loop_parameters_with_initialization_model(TargetT& parametric_gradient,
                                                              TargetT& current_estimate,
                                                              DynamicDiscretisedDensity& dyn_image_estimate,
                                                              DynamicDiscretisedDensity& dyn_image_reference_data,
                                                              DynamicDiscretisedDensity& dyn_image_nested_loop_estimate);

  // The principal nested EM reconstruction method that performs the generalized Patlak reconstruction
  // It can be used either (1) for EM initialization, together with standard Patlak EM method, or (2) for regular nested EM
  // reconstruction.
  virtual void estimate_nested_loop_parameters_with_model(DynamicDiscretisedDensity& impulse_response_gradient,
                                                          TargetT& current_estimate,
                                                          DynamicDiscretisedDensity& dyn_image_estimate,
                                                          DynamicDiscretisedDensity& dyn_image_reference_data,
                                                          DynamicDiscretisedDensity& dyn_image_nested_loop_estimate);

  double actual_compute_objective_function_without_penalty(const TargetT& current_estimate, const int subset_num) override;

  Succeeded set_up_before_sensitivity(shared_ptr<const TargetT> const& target_sptr) override;

  //! Add subset sensitivity to existing data
  /*! \todo Current implementation does NOT add to the subset sensitivity, but overwrites
   */
  void add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const override;

  //! Add subset initialization sensitivity to existing initialization data
  /*! \todo Current implementation does NOT add to the subset sensitivity, but overwrites
   */
  virtual void add_subset_initialization_sensitivity(TargetT& initialization_sensitivity, const int subset_num) const;

  Succeeded actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                                   const TargetT& input,
                                                                                   const int subset_num) const override;

public:
  /*! \name Functions to get parameters
   \warning Be careful with changing shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  const DynamicProjData& get_dyn_proj_data() const;
  const shared_ptr<DynamicProjData>& get_dyn_proj_data_sptr() const;
  const int get_max_segment_num_to_process() const;
  const bool get_zero_seg0_end_planes() const;
  const DynamicProjData& get_input_data() const override;
  const DynamicProjData& get_additive_dyn_proj_data() const;
  const shared_ptr<DynamicProjData>& get_additive_dyn_proj_data_sptr() const;
  const ProjectorByBinPair& get_projector_pair() const;
  const shared_ptr<ProjectorByBinPair>& get_projector_pair_sptr() const;
  const BinNormalisation& get_normalisation() const;
  const shared_ptr<BinNormalisation>& get_normalisation_sptr() const;
  const DynamicDiscretisedDensity get_model_sensitivity_impulse_response() const;
  const TargetT& get_initialization_model_sensitivity_image() const;
  const shared_ptr<TargetT>& get_initialization_model_sensitivity_image_sptr() const;
  //@}

  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with setting shared pointers. If you modify the objects in
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  void set_recompute_sensitivity(const bool);
  void set_sensitivity_sptr(const shared_ptr<TargetT>&);
  int set_num_subsets(const int num_subsets) override;
  void set_input_data(const shared_ptr<ExamData>&) override;
  void set_additive_proj_data_sptr(const shared_ptr<ExamData>&) override;
  void set_normalisation_sptr(const shared_ptr<BinNormalisation>&) override;
  //@}
protected:
  //! Filename with input projection data
  std::string _input_filename;

  //! points to the object for the total input projection data
  shared_ptr<DynamicProjData> _dyn_proj_data_sptr;

  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int _max_segment_num_to_process;

  /**********************/
  ParseAndCreateFrom<TargetT, DynamicProjData> target_parameter_parser;

  /**********************/
  //! the current subiteration index in the nested loop for EM estimation
  //  using the generalized Patlak model, i.e. GeneralizedPatlakPlot
  int nested_subiterations_num;

  //! the current subiteration index in the nested loop for EM estimation
  //  using the standard Patlak model, i.e. PatlakPlot
  // (for the purpose of proper initialization of the generalized Patlak model estimates)
  int nested_initialization_subiterations_num;

  //! restrict updates (larger nested relative updates will be thresholded)
  double maximum_nested_relative_change;

  //! restrict updates (smaller nested relative updates will be thresholded)
  double minimum_nested_relative_change;

  //! Boolean value to determine whether the current global subiteration is performed under initialization mode
  bool is_initialization_subiteration;

  //! Boolean value to determine whether the initialization mode will alternate between
  //  standard (first) and generalized (second) Patlak nested reconstruction within each initialization iteration
  bool is_alternating_initialization_model;

  /********************************/
  //! name of file in which additive projection data are stored
  std::string _additive_dyn_proj_data_filename;
  //! points to the additive projection data
  /*! the projection data in this file is bin-wise added to forward projection results*/
  shared_ptr<DynamicProjData> _additive_dyn_proj_data_sptr;
  /*! the normalisation or/and attenuation data */
  shared_ptr<BinNormalisation> _normalisation_sptr;
  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> _projector_pair_ptr;
  //! signals whether to zero the data in the end planes of the projection data
  bool _zero_seg0_end_planes;

  // Patlak Plot Parameters
  /*! the generalizedPatlak plot pointer where all the parameters are stored */
  shared_ptr<GeneralizedPatlakPlot> _patlak_plot_sptr;

  //! dynamic image template
  DynamicDiscretisedDensity _dyn_image_template;
  DynamicDiscretisedDensity _imp_response_image_template;

  BasicCoordinate<2, int> model_array_min, model_array_max;

  // Define a vector for the GenralizedPatlak model sensitivity impulse response
  DynamicDiscretisedDensity sensitivity_impulse_response_image;
  DynamicDiscretisedDensity impulse_response;

  // Define a shared pointer for the (linear kinetic) model sensitivity image
  shared_ptr<TargetT> initialization_model_sensitivity_image_sptr;

  void compute_initialization_model_sensitivity_image(const TargetT& param_image);

  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const override;

  void compute_model_sensitivity_impulse_response(DynamicDiscretisedDensity& impulse_reponse);

  void add_subset_impulse_response_sensitivity(DynamicDiscretisedDensity& impulse_response, const int subset_num) const;

  //! Sets defaults for parsing
  /*! Resets \c sensitivity_filename and \c sensitivity_sptr and
     \c recompute_sensitivity to \c false.
  */
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonNestedLogLikelihoodWithGeneralizedPatlakAndDynamicProjectionData.inl"

#endif