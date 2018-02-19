//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd

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
  \brief Declaration of class 
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin

  \author Kris Thielemans
  \author Sanida Mustafovic

*/
#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h" 
#include "stir/ProjDataInMemory.h"

#include "stir/ExamInfo.h"
START_NAMESPACE_STIR


/*!
  \ingroup GeneralisedObjectiveFunction
  \brief Class for PET list mode data from static images for a scanner with discrete detectors.

  If the scanner has discrete (and stationary) detectors, it can be modeled via  ProjMatrixByBin and BinNormalisation.
  
  \see PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  If the list mode data is binned (with LmToProjData) without merging
  any bins, then the log likelihood computed from list mode data and
  projection data will be identical.
*/

template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin: 
public RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT> >

{

private:
typedef RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>,
                                GeneralisedObjectiveFunction<TargetT>,
                                PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT> > 
        base_type;

public:
 
 //! Name which will be used when parsing a GeneralisedObjectiveFunction object 
  static const char * const registered_name; 
  
  PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>(); 

  //! This should compute the gradient of the objective function at the  current_image_estimate
  virtual  
  void compute_sub_gradient_without_penalty_plus_sensitivity(TargetT& gradient,  
                                         const TargetT &current_estimate,  
                                         const int subset_num);  
  virtual TargetT * construct_target_ptr() const;  

  int set_num_subsets(const int new_num_subsets);

protected:
  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                      const int subset_num)
  { // TODO 
    error("compute_objective_function_without_penalty Not implemented yet");
    return 0; 
  }

  virtual Succeeded 
    set_up_before_sensitivity(shared_ptr <TargetT > const& target_sptr); 
 
  // TODO
  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const
  {
    error("add_subset_sensitivity not implemented yet");
  }
  
  //! Maximum ring difference to take into account
  /*! \todo Might be removed */
  int  max_ring_difference_num_to_process;

  //! Stores the projectors that are used for the computations
  shared_ptr<ProjMatrixByBin> PM_sptr;
  //shared_ptr<ProjectorByBinPair> projector_by_bin_pair;
  
  //! points to the additive projection data
  shared_ptr<ProjDataInMemory> additive_proj_data_sptr; 
 
  std::string additive_projection_data_filename ; 
  //! ProjDataInfo
  shared_ptr<ProjDataInfo> proj_data_info_cyl_uncompressed_ptr; 

  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets keys
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();
  virtual bool post_processing();

  virtual bool actual_subsets_are_approximately_balanced(std::string& warning_message) const;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
