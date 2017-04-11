/*
    Copyright (C) 2003 - 2011-02-23, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  \author Kris Thielemans
  \author Sanida Mustafovic

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndProjData_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndProjData_H__

#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
//#include "stir/ProjData.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
//#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/TimeFrameDefinitions.h"
#ifdef STIR_MPI
#include "stir/recon_buildblock/distributable.h" // for  RPC_process_related_viewgrams_type
#endif

START_NAMESPACE_STIR

class DistributedCachingInformation;

//#ifdef STIR_MPI_CLASS_DEFINITION
//#define PoissonLogLikelihoodWithLinearModelForMeanAndProjData PoissonLogLikelihoodWithLinearModelForMeanAndProjData_MPI
//#endif



/*!
  \ingroup GeneralisedObjectiveFunction
  \brief An objective function class appropriate for PET emission data

  Measured data is given by a ProjData object, and the linear operations
  necessary for computing the gradient of the objective function
  are performed via a ProjectorByBinPair object together with
  a BinNormalisation object.

  \see PoissonLogLikelihoodWithLinearModelForMean for notation.

  Often, the probability matrix \f$P\f$ can be written as the product
  of a diagonal matrix \f$D\f$ and another matrix \f$F\f$
  \f[ P = D G \f]
  The measurement model can then be written as

  \f[ P \lambda + r = D ( F \lambda + a ) \f]

  and backprojection is obviously

  \f[ P' y = F' D y \f]

  This can be generalised by using a different matrix \f$B\f$ to perform
  the backprojection.  

  The expression for the gradient becomes

  \f[ B D \left[ y / \left( D (F \lambda + a) \right) \right] - B D 1  =
      B  \left[ y / \left(F \lambda + a \right) \right] - B D 1
  \f]

  where \f$D\f$ dropped from the first term. This was probably noticed
  for the first time in the context of MLEM by Hebert and Leahy, but
  is in fact a feature of the gradient (and indeed log-likelihood).
  This was first brought to Kris Thielemans' attention by
  Michael Zibulevsky and Arkady Nemirovsky.

  Note that if \f$F\f$ and \f$B\f$ are not exact transposed operations, the
  <i>gradient</i> does no longer correspond to the gradient
  of a well-defined function.

  In this class, the operation of multiplying with \f$F\f$ is performed
  by the forward projector, and with \f$B\f$ by the back projector.
  Multiplying with \f$D\f$ corresponds to the BinNormalisation::undo() 
  function. Finally, the background term \f$a\f$ is stored in
  \c additive_proj_data_sptr (note: this is <strong>not</strong> \f$r\f$).

  \par Parameters for parsing

  \verbatim
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters:=

  ; emission projection data
  input file :=

  maximum absolute segment number to process :=
  zero end planes of segment 0 :=

  ; see ProjectorByBinPair hierarchy for possible values
  Projector pair type :=
  
  ; reserved value: 0 means none
  ; see PoissonLogLikelihoodWithLinearModelForMeanAndProjData 
  ; class documentation
  additive sinogram :=

  ; normalisation (and attenuation correction)
  ; time info can be used for dead-time correction

  ; see TimeFrameDefinitions
  time frame definition filename :=
  time frame number :=
  ; see BinNormalisation hierarchy for possible values
  Bin Normalisation type :=

  End PoissonLogLikelihoodWithLinearModelForMeanAndProjData Parameters :=
  \endverbatim
*/
template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndProjData: 
public  RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
{
 private:
  typedef RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
    base_type;

public:

  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char * const registered_name; 

  //
  /*! \name Variables for STIR_MPI

  Only used when STIR_MPI is enabled. 
  \todo move to protected area
  */
  //@{
  //! points to the information object needed to support distributed caching
  DistributedCachingInformation * caching_info_ptr;
  //#ifdef STIR_MPI
  //!enable/disable key for distributed caching 
  bool distributed_cache_enabled;
  bool distributed_tests_enabled;
  bool message_timings_enabled;
  double message_timings_threshold;
  bool rpc_timings_enabled;
  //#endif
  //@}


  //! Default constructor calls set_defaults()
  PoissonLogLikelihoodWithLinearModelForMeanAndProjData();

  //! Destructor
  /*! Calls end_distributable_computation()
   */
  ~PoissonLogLikelihoodWithLinearModelForMeanAndProjData();

  //! Returns a pointer to a newly allocated target object (with 0 data).
  /*! Dimensions etc are set from the \a proj_data_sptr and other information set by parsing,
    such as \c zoom, \c output_image_size_z etc.
  */
  virtual TargetT *
    construct_target_ptr() const; 

  /*! \name Functions to get parameters
   \warning Be careful with changing shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.
  */
  //@{
  const ProjData& get_proj_data() const;
  const shared_ptr<ProjData>& get_proj_data_sptr() const;
  const int get_max_segment_num_to_process() const;
  const int get_max_timing_pos_num_to_process() const;
  const bool get_zero_seg0_end_planes() const;
  const ProjData& get_additive_proj_data() const;
  const shared_ptr<ProjData>& get_additive_proj_data_sptr() const;
  const ProjectorByBinPair& get_projector_pair() const;
  const shared_ptr<ProjectorByBinPair>& get_projector_pair_sptr() const;
  const int get_time_frame_num() const;
  const TimeFrameDefinitions& get_time_frame_definitions() const;
  const BinNormalisation& get_normalisation() const;
  const shared_ptr<BinNormalisation>& get_normalisation_sptr() const;
  //@}
  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with changing shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.

  */
  //@{
  int set_num_subsets(const int num_subsets);
  void set_proj_data_sptr(const shared_ptr<ProjData>&);
  void set_max_segment_num_to_process(const int);
  void set_max_timing_pos_num_to_process(const int);
  void set_zero_seg0_end_planes(const bool);
  //N.E. Changed to ExamData
  virtual void set_additive_proj_data_sptr(const shared_ptr<ExamData>&);
  void set_projector_pair_sptr(const shared_ptr<ProjectorByBinPair>&) ;
  void set_frame_num(const int);
  void set_frame_definitions(const TimeFrameDefinitions&);
  virtual void set_normalisation_sptr(const shared_ptr<BinNormalisation>&);

  virtual void set_input_data(const shared_ptr<ExamData> &);
  //@}
  
  virtual void 
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
                                                          const TargetT &current_estimate, 
                                                          const int subset_num); 

#if 0
  // currently not used
  float sum_projection_data() const;
#endif
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const;

 protected:
  virtual Succeeded 
    set_up_before_sensitivity(shared_ptr <TargetT > const& target_sptr);

  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                      const int subset_num);

  /*!
    The Hessian (without penalty) is approximatelly given by:
    \f[ H_{jk} = - \sum_i P_{ij} h_i^{''}(y_i) P_{ik} \f]
    where
    \f[ h_i(l) = y_i log (l) - l; h_i^{''}(y_i) ~= -1/y_i; \f]
    and \f$P_{ij} \f$ is the probability matrix. 
    Hence
    \f[ H_{jk} =  \sum_i P_{ij}(1/y_i) P_{ik} \f]

    In the above, we've used the plug-in approximation by replacing 
    forward projection of the true image by the measured data. However, the
    later are noisy and this can create problems. 

    \todo Two work-arounds for the noisy estimate of the Hessian are listed below, 
    but they are currently not implemented.

    One could smooth the data before performing the quotient. This should be done 
    after normalisation to avoid problems with the high-frequency components in 
    the normalisation factors:
    \f[ H_{jk} =  \sum_i G_{ij}{1 \over n_i \mathrm{smooth}( n_i y_i)} G_{ik} \f]
    where the probability matrix is factorised in a detection efficiency part (i.e. the
    normalisation factors \f$n_i\f$) times a geometric part:
    \f[ P_{ij} = {1 \over n_i } G_{ij}\f]

    It has also been suggested to use \f$1 \over y_i+1 \f$ (at least if the data are still Poisson.
  */
  virtual Succeeded 
      actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                             const TargetT& input,
                                                                             const int subset_num) const;

protected:
  //! Filename with input projection data
  std::string input_filename;

  //! points to the object for the total input projection data
  shared_ptr<ProjData> proj_data_sptr;

  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

  //! the maximum absolute time-of-flight bin number to use in the reconstruction
  /*! convention: if -1, use get_max_tof_pos_num()*/
  int max_timing_pos_num_to_process;

  /**********************/
  // image stuff
  // TODO to be replaced with single class or so (TargetT obviously)
  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int output_image_size_xy; // KT 10122001 appended _xy

  //! the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int output_image_size_z; // KT 10122001 new

  //! the zoom factor
  double zoom;

  //! offset in the x-direction
  double Xoffset;

  //! offset in the y-direction
  double Yoffset;

  // KT 20/06/2001 new
  //! offset in the z-direction
  double Zoffset;
  /********************************/


  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> projector_pair_ptr;

  //! Backprojector used for sensitivity computation
  shared_ptr<BackProjectorByBin> sens_backprojector_sptr;

  //! signals whether to zero the data in the end planes of the projection data
  bool zero_seg0_end_planes;

  //! name of file in which additive projection data are stored
  std::string additive_projection_data_filename;


  shared_ptr<ProjData> additive_proj_data_sptr;

  shared_ptr<BinNormalisation> normalisation_sptr;

 // TODO doc
  int frame_num;
  std::string frame_definition_filename;
  TimeFrameDefinitions frame_defs;

//Loglikelihood computation parameters
 // TODO rename and move higher up in the hierarchy 
  //! subiteration interval at which the loglikelihood function is evaluated
  int loglikelihood_computation_interval;

  //! indicates whether to evaluate the loglikelihood function for all bins or the current subset
  bool compute_total_loglikelihood;

  //! name of file in which loglikelihood measurements are stored
  std::string loglikelihood_data_filename;

  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets keys for parsing
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();
  //! checks values after parsing
  /*! Has to be called by post_processing in the leaf-class */  
  virtual bool post_processing();

  //! Checks of the current subset scheme is approximately balanced
  /*! For this class, this means that the sub-sensitivities are 
      approximately the same. The test simply looks at the number
      of views etc. It ignores unbalancing caused by normalisation_sptr
      (e.g. for instance when using asymmetric attenuation).
  */
  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const;
 private:
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr;

  void
	  add_view_seg_to_sensitivity(TargetT& sensitivity, const ViewSegmentNumbers& view_seg_nums) const;
};

#ifdef STIR_MPI
//made available to be called from DistributedWorker object
RPC_process_related_viewgrams_type RPC_process_related_viewgrams_gradient;
RPC_process_related_viewgrams_type RPC_process_related_viewgrams_accumulate_loglikelihood;
#endif

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
