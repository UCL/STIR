//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2018, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Declaration of class 
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeData

  \author Kris Thielemans
  \author Sanida Mustafovic

*/

#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeData_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeData_H__


//#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.h"
#include "stir/listmode/ListModeData.h"
#include "stir/ParseAndCreateFrom.h"
#include "stir/TimeFrameDefinitions.h"

START_NAMESPACE_STIR

/*!
  \ingroup GeneralisedObjectiveFunction
  \brief An objective function class appropriate for PET list mode data

  The statistics for list mode data is slightly different from having a set
  of counts, see the paper
  H. H. Barrett, L. Parra, and T. White, 
  <i>List-mode likelihood,</i> J. Optical Soc. Amer. A, vol. 14, no. 11, 1997.

  However, it is intuitive that the list mode likelihood can be
  derived from that 'binned' likelihood by taking smaller and smaller
  time bins. One can then see that the gradient of the list mode
  likelihood can be computed similar to the 'binned' case, but now
  with a sum over events. The sensitivity still needs a sum (or integral)
  over all possible detections.

  At present, STIR does not contain any classes to forward/back
  project list mode data explictly. So, the sum over events cannot
  be computed for arbitrary list mode data. This is currently
  done in derived classes.

  For list mode reconstructions, computing the sensitivity is sometimes
  conceptually very difficult (how to compute the integral) or impractical.
  So, even if we will be abel to compute the sum over events in a generic way,
  the add_subset_sensitivity() function will have to be implemented by
  a derived class, specific for the measurement.

*/
template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndListModeData: 
public PoissonLogLikelihoodWithLinearModelForMean<TargetT> 
{
private: 

  typedef PoissonLogLikelihoodWithLinearModelForMean<TargetT> base_type;

public:   

 
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeData();  
 
  //virtual TargetT * construct_target_ptr();  
 
  virtual Succeeded
   set_up(shared_ptr <TargetT > const& target_sptr);
 
  virtual  
  void compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,  
	    				 const TargetT &current_estimate,  
					 const int subset_num)=0;  
  //! time frame definitions
  /*! \todo This is currently used to be able to compute the gradient for 
    one time frame. However, it probably does not belong here.
    For instance when fitting kinetic model parameters from list
    mode data, time frames are in principle irrelevant. So, we will
    probably shift this to the derived class.
  */
    TimeFrameDefinitions frame_defs;

    virtual void set_normalisation_sptr(const shared_ptr<BinNormalisation>&);
    virtual void set_additive_proj_data_sptr(const shared_ptr<ExamData>&);

    virtual void set_input_data(const shared_ptr<ExamData> &);
    virtual const ListModeData& get_input_data() const;
protected:
  std::string frame_defs_filename;

  //! Filename with input projection data
  std::string list_mode_filename;

  shared_ptr<ProjData> additive_proj_data_sptr;

  shared_ptr<BinNormalisation> normalisation_sptr;
 
  //! Listmode pointer
  shared_ptr<ListModeData> list_mode_data_sptr;
 
  unsigned int current_frame_num;

  //! This is part of some functionality I transfer from LmToProjData.
  long int num_events_to_use;
   //! Reconstruct based on time frames
   bool do_time_frame;
 
  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets keys
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap(); 

  virtual bool post_processing(); 

   //! will be called when a new time frame starts
   /*! The frame numbers start from 1. */
   virtual void start_new_time_frame(const unsigned int new_frame_num);

   ParseAndCreateFrom<TargetT, ListModeData> target_parameter_parser;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
