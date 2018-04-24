//
//
/*
    Copyright (C) 2006- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London

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
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Charalampos Tsoumpas

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData_H__
#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/VectorWithOffset.h"
#include "stir/DynamicProjData.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/modelling/KineticParameters.h"
#include "stir/modelling/PatlakPlot.h"

START_NAMESPACE_STIR

/*!
  \ingroup GeneralisedObjectiveFunction
  \ingroup modelling
  \brief a base class for LogLikelihood of independent Poisson variables 
  where the mean values are linear combinations of the kinetic parameters.

  \par Parameters for parsing

*/

template <typename TargetT>
class PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData: 
public  RegisteredParsingObject<PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
{
 private:
  typedef  RegisteredParsingObject<PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> > base_type;
  typedef PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float> > SingleFrameObjFunc ;
  VectorWithOffset<SingleFrameObjFunc> _single_frame_obj_funcs;
 public:
  
  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char * const registered_name; 

  PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData();

  //! Returns a pointer to a newly allocated target object (with 0 data).
  /*! Dimensions etc are set from the \a dyn_proj_data_sptr and other information set by parsing,
    such as \c zoom, \c output_image_size_z etc.
  */
  virtual TargetT *
    construct_target_ptr() const; 
  virtual void 
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
                                                          const TargetT &current_estimate, 
                                                          const int subset_num); 

 protected:
  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                      const int subset_num);

  virtual Succeeded set_up_before_sensitivity(shared_ptr <TargetT> const& target_sptr);

  //! Add subset sensitivity to existing data
  /*! \todo Current implementation does NOT add to the subset sensitivity, but overwrites
   */
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const;

  virtual Succeeded 
      actual_add_multiplication_with_approximate_sub_Hessian_without_penalty(TargetT& output,
                                                                             const TargetT& input,
                                                                             const int subset_num) const;
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
  const DynamicProjData& get_additive_dyn_proj_data() const;
  const shared_ptr<DynamicProjData>& get_additive_dyn_proj_data_sptr() const;
  const ProjectorByBinPair& get_projector_pair() const;
  const shared_ptr<ProjectorByBinPair>& get_projector_pair_sptr() const;
  const BinNormalisation& get_normalisation() const;
  const shared_ptr<BinNormalisation>& get_normalisation_sptr() const;
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
  virtual int set_num_subsets(const int num_subsets);

  virtual void set_normalisation_sptr(const shared_ptr<BinNormalisation>&);
  virtual void set_additive_proj_data_sptr(const shared_ptr<ExamData>&);

  virtual void set_input_data(const shared_ptr<ExamData> &);
  virtual const DynamicProjData& get_input_data() const;
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
  // image stuff
  // TODO to be replaced with single class or so (TargetT obviously)
  //! the output image size in x and y direction
  /*! convention: if -1, use a size such that the whole FOV is covered
  */
  int _output_image_size_xy; // KT 10122001 appended _xy

  //! the output image size in z direction
  /*! convention: if -1, use default as provided by VoxelsOnCartesianGrid constructor
  */
  int _output_image_size_z; // KT 10122001 new

  //! the zoom factor
  double _zoom;

  //! offset in the x-direction
  double _Xoffset;

  //! offset in the y-direction
  double _Yoffset;

  // KT 20/06/2001 new
  //! offset in the z-direction
  double _Zoffset;

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
  /*! the patlak plot pointer where all the parameters are stored */
  shared_ptr<PatlakPlot> _patlak_plot_sptr;
  //! dynamic image template
  DynamicDiscretisedDensity _dyn_image_template;

  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const;

  //! Sets defaults for parsing 
  /*! Resets \c sensitivity_filename and \c sensitivity_sptr and
     \c recompute_sensitivity to \c false.
  */
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearKineticModelAndDynamicProjectionData.inl"

#endif
