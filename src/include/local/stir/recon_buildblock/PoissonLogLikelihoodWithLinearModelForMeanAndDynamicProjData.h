//
//
/*
    Copyright (C) 2006- 2011, Hammersmith Imanet Ltd

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
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Charalampos Tsoumpas

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData_H__

#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/VectorWithOffset.h"
#include "stir/DynamicProjData.h"
#include "stir/DynamicDiscretisedDensity.h"

START_NAMESPACE_STIR
// ChT::ToDo: If this class appears to be useful I have to define some of the functions that are not specifically defined, yet. 

/*!
  \ingroup GeneralisedObjectiveFunction
  \brief a base class for LogLikelihood of independent Poisson variables 
  where the mean values are linear combinations of the frames.

  \par Parameters for parsing

*/

template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData: 
public  RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>,
				GeneralisedObjectiveFunction<TargetT>,
				PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
{
 private:
  typedef  RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData<TargetT>,
				GeneralisedObjectiveFunction<TargetT>,
				PoissonLogLikelihoodWithLinearModelForMean<TargetT> > base_type;
  typedef PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float> > SingleFrameObjFunc ;
  VectorWithOffset<SingleFrameObjFunc> _single_frame_obj_funcs;
 public:
  
  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char * const registered_name; 
#if 0 // ChT::ToDo
  PoissonLogLikelihoodWithLinearModelForMeanAndDynamicProjData();

  //! Returns a pointer to a newly allocated target object (with 0 data).
  /*! Dimensions etc are set from the \a proj_data_sptr and other information set by parsing,
    such as \c zoom, \c output_image_size_z etc.
  */
  virtual TargetT *
    construct_target_ptr() const; 

  virtual void 
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient, 
							  const TargetT &current_estimate, 
							  const int subset_num); 

  virtual float 
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
						      const int subset_num);

  virtual Succeeded set_up_before_sensitivity(shared_ptr <TargetT> const& target_sptr);

  //! Add subset sensitivity to existing data
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const;

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
  //@}
#endif // ChT::ToDo
 protected:
  //! Filename with input dynamic projection data
  string _input_filename;

  //! points to the object for the total input dynamic projection data
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

  TimeFrameDefinitions _frame_defs;
  
  /********************************/
  //! name of file in which additive projection data are stored
  string _additive_dyn_proj_data_filename;
 //! points to the additive projection data
  /*! the projection data in this file is bin-wise added to forward projection results*/
  shared_ptr<DynamicProjData> _additive_dyn_proj_data_sptr;
  /*! the normalisation or/and attenuation data */
  shared_ptr<BinNormalisation> _normalisation_sptr;
  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> _projector_pair_ptr;
  //! signals whether to zero the data in the end planes of the projection data
  bool _zero_seg0_end_planes;

  bool actual_subsets_are_approximately_balanced(string& warning_message) const;

  //! Sets defaults for parsing 
  /*! Resets \c sensitivity_filename and \c sensitivity_sptr and
     \c recompute_sensitivity to \c false.
  */
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_STIR

#endif
