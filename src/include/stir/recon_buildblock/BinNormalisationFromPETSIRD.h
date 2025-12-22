//
//
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisationFromPETSIRD

  \author Nikos Efthimiou
*/
/*
    Copyright (C) 2025, UMCG
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_recon_buildblock_BinNormalisationFromPETSIRD_H__
#define __stir_recon_buildblock_BinNormalisationFromPETSIRD_H__

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/recon_buildblock/BinNormalisationWithCalibration.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/shared_ptr.h"
#include "stir/PETSIRDInfo.h"

using std::string;

START_NAMESPACE_STIR

class BinNormalisationFromPETSIRD
    : public RegisteredParsingObject<BinNormalisationFromPETSIRD, BinNormalisation, BinNormalisationWithCalibration>
{
private:
  using base_type = BinNormalisationWithCalibration;

public:
  //! Name which will be used when parsing a BinNormalisation object
  static const char* const registered_name;

  BinNormalisationFromPETSIRD();

  BinNormalisationFromPETSIRD(const std::string& filename);

  Succeeded set_up(const shared_ptr<const ExamInfo>& exam_info_sptr, const shared_ptr<const ProjDataInfo>&) override;

  float get_uncalibrated_bin_efficiency(const Bin& bin) const override;

  inline bool with_detector_efficiencies() const { return m_with_detector_efficiencies; }
  inline bool with_dead_time() const { return m_with_dead_time; }
  inline bool with_geometric_factors() const { return m_with_geometric_factors; }

private:
  void set_defaults() override;

  void initialise_keymap() override;

  bool post_processing() override;

  void read_norm_data(const string& filename);

  string normalisation_filename;

  //! Flag to enable/disable detector efficiency
  bool m_with_detector_efficiencies;
  //! Flag to enable/disable dead time correction
  bool m_with_dead_time;
  //! Flag to enable/disable geometric factors
  bool m_with_geometric_factors;
  //   shared_ptr<PETSIRDInfo> petsird_info_sptr;
  shared_ptr<petsird::PETSIRDReaderBase> petsird_data_sptr;

  shared_ptr<petsird::ScannerInformation> scanner_info_sptr;

  shared_ptr<const PETSIRDInfo> petsird_info_sptr;
};

END_NAMESPACE_STIR

#endif
