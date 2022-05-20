//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2015, Univ. of Leeds
    Copyright (C) 2016, UCL
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GeneralisedObjectiveFunction
  \brief Declaration of class
  stir::PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Sanida Mustafovic

*/
#ifndef __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin_H__
#define __stir_recon_buildblock_PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndListModeData.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInMemory.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#include "stir/ExamInfo.h"
#include "stir/recon_buildblock/distributable.h"
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

  //! Computes the gradient of the objective function at the \a current_estimate overwriting \a gradient.
  /*!
   \warning If <code>add_sensitivity = false</code> and <code>use_subset_sensitivities = false</code> will return an error
   because the gradient will not be correct. Try <code>use_subset_sensitivities = true</code>.
   */
    virtual
    void actual_compute_subset_gradient_without_penalty(TargetT& gradient,
                                                        const TargetT &current_estimate,
                                                        const int subset_num,
                                                        const bool add_sensitivity);

  virtual TargetT * construct_target_ptr() const;

  int set_num_subsets(const int new_num_subsets);

  const shared_ptr<BinNormalisation> &
  get_normalisation_sptr() const
  { return this->normalisation_sptr; }

  virtual unique_ptr<ExamInfo> get_exam_info_uptr_for_target() const;

  void set_proj_matrix(const shared_ptr<ProjMatrixByBin>&);

  void set_proj_data_info(const ProjData& arg);

  void set_skip_balanced_subsets(const bool arg);

  void set_max_ring_difference(const int arg);


protected:
  virtual double
    actual_compute_objective_function_without_penalty(const TargetT& current_estimate,
                                                      const int subset_num)
  { // TODO
    error("compute_objective_function_without_penalty Not implemented yet");
    return 0;
  }

  virtual Succeeded
    set_up_before_sensitivity(shared_ptr <const TargetT > const& target_sptr);

  virtual void
    add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const;

  //! This function caches the listmode file. It is run during post-processing.
  Succeeded cache_listmode_file();

  //! Maximum ring difference to take into account
  /*! \todo Might be removed */
  int  max_ring_difference_num_to_process;

  //! Stores the projectors that are used for the computations
  shared_ptr<ProjMatrixByBin> PM_sptr;

  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> projector_pair_sptr;

  //! points to the additive projection data
  shared_ptr<ProjData> additive_proj_data_sptr;

  std::string additive_projection_data_filename ;
  //! ProjDataInfo
  shared_ptr<ProjDataInfo> proj_data_info_sptr;

  //! sets any default values
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets keys
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();
  virtual bool post_processing();

  virtual bool actual_subsets_are_approximately_balanced(std::string& warning_message) const;

  void
    add_view_seg_to_sensitivity(const ViewSegmentNumbers& view_seg_nums) const;

  //! Cache of the listmode file
  std::vector<BinAndCorr>  record_cache;
  //! The additive sinogram will not be read in memory
  bool reduce_memory_usage;
  //! If you know, or have previously checked that the number of subsets is balanced for your
  //! Scanner geometry, you can skip future checks.
  bool skip_balanced_subsets;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
