/*
    Copyright (C) 2025, UMCG
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details

/*!
  \file
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationFromPETSIRD

  \author Nikos Efthimiou
*/

#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"
#include "stir/recon_buildblock/BinNormalisationFromPETSIRD.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

START_NAMESPACE_STIR

const char* const BinNormalisationFromPETSIRD::registered_name = "From PETSIRD";

void
BinNormalisationFromPETSIRD::set_defaults()
{
  base_type::set_defaults();
  normalisation_filename = "";
  m_with_detector_efficiencies = true;
  m_with_dead_time = true;
  m_with_geometric_factors = true;
}

void
BinNormalisationFromPETSIRD::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Bin Normalisation From PETSIRD");
  parser.add_key("normalisation_filename", &normalisation_filename);
  parser.add_stop_key("End Bin Normalisation From PETSIRD");
}

bool
BinNormalisationFromPETSIRD::post_processing()
{
  if (base_type::post_processing())
    return true;
  read_norm_data(normalisation_filename);
  return false;
}

BinNormalisationFromPETSIRD::BinNormalisationFromPETSIRD()
{
  set_defaults();
}

BinNormalisationFromPETSIRD::BinNormalisationFromPETSIRD(const std::string& filename)
{
  read_norm_data(filename);
}

float
BinNormalisationFromPETSIRD::get_uncalibrated_bin_efficiency(const Bin& bin) const
{

  DetectionPositionPair<> dp;

  if (const auto* proj_cyl = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get()))
    {
      proj_cyl->get_det_pos_pair_for_bin(dp, bin);
    }
  else if (const auto* proj_blk = dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(proj_data_info_sptr.get()))
    {
      proj_blk->get_det_pos_pair_for_bin(dp, bin);
    }
  else
    {
      error("BinNormalisationFromPETSIRD: ProjDataInfo is neither Cylindrical nor BlocksOnCylindrical");
    }

  return petsird_info_sptr->get_detection_efficiency_for_bin(dp);
}

Succeeded
BinNormalisationFromPETSIRD::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,
                                    const shared_ptr<const ProjDataInfo>& proj_data_info_ptr_v)
{
  base_type::set_up(exam_info_sptr, proj_data_info_ptr_v);

  return Succeeded::yes;
}

void
BinNormalisationFromPETSIRD::read_norm_data(const string& filename)
{
  petsird::Header header;
  petsird_data_sptr.reset(new petsird::binary::PETSIRDReader(filename));

  petsird_data_sptr->ReadHeader(header);

  petsird_info_sptr = std::make_shared<PETSIRDInfo>(header);
}

END_NAMESPACE_STIR