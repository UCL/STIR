//
//
/*
    Copyright (C) 2005- 2011, Hammersmith Imanet Ltd
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
  \ingroup recon_buildblock
  \brief Declaration of class stir::PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion

  \author Kris Thielemans

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#include "stir/GatedProjData.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/DataProcessor.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/recon_buildblock/SumOfGeneralisedObjectiveFunctions.h"
#include "stir/TimeFrameDefinitions.h"

START_NAMESPACE_STIR
class GatedProjData;

/*!
  \ingroup recon_buildblock
  \brief
*/
template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion 
: public  RegisteredParsingObject
            <PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>,
             GeneralisedObjectiveFunction<TargetT>,
             SumOfGeneralisedObjectiveFunctions<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                                TargetT, 
                                                PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
>
{
 private:
  typedef
  RegisteredParsingObject
    <PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion<TargetT>,
    GeneralisedObjectiveFunction<TargetT>,
    SumOfGeneralisedObjectiveFunctions<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<TargetT>,
                                       TargetT, 
                                       PoissonLogLikelihoodWithLinearModelForMean<TargetT> >
    >
    base_type;

public:

  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char * const registered_name; 

  //! Default constructor calls set_defaults()
  PoissonLogLikelihoodWithLinearModelForMeanAndGatedProjDataWithMotion();
  
  /*! \name Functions to set parameters
    This can be used as alternative to the parsing mechanism.
   \warning After using any of these, you have to call set_up().
   \warning Be careful with setting shared pointers. If you modify the objects in 
   one place, all objects that use the shared pointer will be affected.

  */
  //@{
  void set_proj_data_sptr(const shared_ptr<GatedProjData>&);
  void set_max_segment_num_to_process(const int);
  void set_zero_seg0_end_planes(const bool);
  void set_additive_proj_data_sptr(const shared_ptr<GatedProjData>&);
  void set_projector_pair_sptr(const shared_ptr<ProjectorByBinPair>&) ;
  void set_frame_num(const int);
  void set_frame_definitions(const TimeFrameDefinitions&);
  //@}

  virtual
  TargetT *
    construct_target_ptr() const; 

  virtual
    void
    compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,
							  const TargetT& target, 
							  const int subset_num);
protected:

  virtual
  Succeeded
    set_up_before_sensitivity(shared_ptr<TargetT > const& target_sptr);

  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const;

  //! Filename with input projection data
  std::string _input_filename;

  //! the maximum absolute ring difference number to use in the reconstruction
  /*! convention: if -1, use get_max_segment_num()*/
  int max_segment_num_to_process;

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

  //! signals whether to zero the data in the end planes of the projection data
  bool zero_seg0_end_planes;

  //! name of file in which loglikelihood measurements are stored
  std::string _additive_projection_data_filename;

 // TODO doc
  int frame_num;
  std::string frame_definition_filename;
  TimeFrameDefinitions frame_defs;
  VectorWithOffset<shared_ptr<BinNormalisation> > _normalisation_sptrs;

 private:
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  shared_ptr<GatedProjData> _gated_proj_data_sptr;
  shared_ptr<GatedProjData> _gated_additive_proj_data_sptr;

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr;

  VectorWithOffset<shared_ptr<DataProcessor<TargetT> > > _forward_transformations;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
