//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2015, Univ. of Leeds
    Copyright (C) 2016, 2022, 2024 UCL
    Copyright (C) 2021, University of Pennsylvania
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
#include "stir/deprecated.h"
#include "stir/recon_buildblock/distributable.h"
#include "stir/error.h"
START_NAMESPACE_STIR

/*!
  \ingroup GeneralisedObjectiveFunction
  \ingroup listmode
  \brief Class for PET list mode data from static images for a scanner with discrete detectors.

  If the scanner has discrete (and stationary) detectors, it can be modeled via  ProjMatrixByBin and BinNormalisation.

  \see PoissonLogLikelihoodWithLinearModelForMeanAndProjData

  If the list mode data is binned (with LmToProjData) without merging
  any bins, then the log likelihood computed from list mode data and
  projection data will be identical.

  Currently, the subset scheme is the same for the projection data and listmode data, i.e.
  based on views. This is suboptimal for listmode data.

  \todo implement a subset scheme based on events
*/

template <typename TargetT>
class PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin
    : public RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>,
                                     GeneralisedObjectiveFunction<TargetT>,
                                     PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>>

{

private:
  typedef RegisteredParsingObject<PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin<TargetT>,
                                  GeneralisedObjectiveFunction<TargetT>,
                                  PoissonLogLikelihoodWithLinearModelForMeanAndListModeData<TargetT>>
      base_type;

public:
  //! Name which will be used when parsing a GeneralisedObjectiveFunction object
  static const char* const registered_name;

  PoissonLogLikelihoodWithLinearModelForMeanAndListModeDataWithProjMatrixByBin();

  //! Computes the gradient of the objective function at the \a current_estimate overwriting \a gradient.
  /*!
   \warning If <code>add_sensitivity = false</code> and <code>use_subset_sensitivities = false</code> will return an error
   because the gradient will not be correct. Try <code>use_subset_sensitivities = true</code>.
   */

  void actual_compute_subset_gradient_without_penalty(TargetT& gradient,
                                                      const TargetT& current_estimate,
                                                      const int subset_num,
                                                      const bool add_sensitivity) override;

  TargetT* construct_target_ptr() const override;

  int set_num_subsets(const int new_num_subsets) override;

  const shared_ptr<BinNormalisation>& get_normalisation_sptr() const { return this->normalisation_sptr; }

  unique_ptr<ExamInfo> get_exam_info_uptr_for_target() const override;

  void set_proj_matrix(const shared_ptr<ProjMatrixByBin>&);

  void set_skip_balanced_subsets(const bool arg);

#if STIR_VERSION < 060000
  STIR_DEPRECATED
  void set_max_ring_difference(const int arg);
#endif

protected:
  /*! \todo this function is not implemented yet and currently calls error() */
  double actual_compute_objective_function_without_penalty(const TargetT& current_estimate, const int subset_num) override
  { // TODO
    error("compute_objective_function_without_penalty Not implemented yet");
    return 0;
  }

  Succeeded set_up_before_sensitivity(shared_ptr<const TargetT> const& target_sptr) override;

  void add_subset_sensitivity(TargetT& sensitivity, const int subset_num) const override;

#if STIR_VERSION < 060000
  //! Maximum ring difference to take into account
  /*! @deprecated */
  int max_ring_difference_num_to_process;
#endif

  //! Triggers calculation of sensitivity using time-of-flight
  bool use_tofsens;

  //! Stores the projectors that are used for the computations
  shared_ptr<ProjMatrixByBin> PM_sptr;

  //! Stores the projectors that are used for the computations
  shared_ptr<ProjectorByBinPair> projector_pair_sptr;

  //! Backprojector used for sensitivity computation
  shared_ptr<BackProjectorByBin> sens_backprojector_sptr;
  //! Proj data info to be used for sensitivity calculations
  /*! This is set to non-TOF data if \c use_tofsens == \c false */
  shared_ptr<ProjDataInfo> sens_proj_data_info_sptr;

  //! sets any default values
  void set_defaults() override;
  //! sets keys for parsing
  void initialise_keymap() override;
  bool post_processing() override;

  bool actual_subsets_are_approximately_balanced(std::string& warning_message) const override;

  //! If you know, or have previously checked that the number of subsets is balanced for your
  //! Scanner geometry, you can skip future checks.
  bool skip_balanced_subsets;

private:
  //! Cache of the current "batch" in the listmode file
  /*! \todo Move this higher-up in the hierarchy as it doesn't depend on ProjMatrixByBin
   */
  std::vector<BinAndCorr> record_cache;

  //! This function loads the next "batch" of data from the listmode file.
  /*!
    This function will either use read_listmode_batch or load_listmode_cache_file.

    \param[in] ibatch the batch number to be read.
    \return \c true if there are no more events to read after this call, \c false otherwise
    \todo Move this function higher-up in the hierarchy as it doesn't depend on ProjMatrixByBin
   */
  bool load_listmode_batch(unsigned int ibatch);

  //! This function reads the next "batch" of data from the listmode file.
  /*!
    This function keeps on reading from the current position in the list-mode data and stores
    prompts events and additive terms in \c record_cache. It also updates \c end_time_per_batch
    such that we know when each batch starts/ends.

    \param[in] ibatch the batch number to be read.
    \return \c true if there are no more events to read after this call, \c false otherwise
    \todo Move this function higher-up in the hierarchy as it doesn't depend on ProjMatrixByBin
    \warning This function has to be called in sequence.
   */
  bool read_listmode_batch(unsigned int ibatch);
  //! This function caches the list-mode batches to file. It is run during set_up()
  /*! \todo Move this function higher-up in the hierarchy as it doesn't depend on ProjMatrixByBin
   */
  Succeeded cache_listmode_file();

  //! Reads the "batch" of data from the cache
  bool load_listmode_cache_file(unsigned int file_id);
  Succeeded write_listmode_cache_file(unsigned int file_id) const;

  unsigned int num_cache_files;
  std::vector<double> end_time_per_batch;
};

END_NAMESPACE_STIR

//#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMean.inl"

#endif
