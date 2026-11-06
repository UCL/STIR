/*
    Copyright (C) 2025, University Medical Center Groningen
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!
  \file BinNormalisationFromPETSIRD.cxx
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationFromPETSIRD

  \author Nikos Efthimiou
*/

#include "petsird/binary/protocols.h"
#include "petsird/hdf5/protocols.h"
#include "stir/recon_buildblock/BinNormalisationFromPETSIRD.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

#include "stir/ExamInfo.h"
#include "stir/Array.h"
#include "stir/IndexRange.h"
#include "stir/IO/read_data.h"
#include "stir/ByteOrder.h"
#include <cmath>

#include "stir/info.h"
#include "stir/format.h"

START_NAMESPACE_STIR

const char* const BinNormalisationFromPETSIRD::registered_name = "From PETSIRD";

void
BinNormalisationFromPETSIRD::set_defaults()
{
  base_type::set_defaults();
  normalisation_filename = "";
  use_hdf5 = false;
  m_with_detector_efficiencies = true;
  m_use_dead_time = true;
  m_with_geometric_factors = true;
}

void
BinNormalisationFromPETSIRD::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Bin Normalisation From PETSIRD");
  parser.add_key("normalisation_filename", &normalisation_filename);
  parser.add_key("use hdf5", &use_hdf5);
  parser.add_key("use_dead_time", &m_use_dead_time);
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
  std::vector<DetectionPositionPair<>> dps;
  if (const auto* proj_cyl = dynamic_cast<const ProjDataInfoCylindricalNoArcCorr*>(proj_data_info_sptr.get()))
    {
      proj_cyl->get_all_det_pos_pairs_for_bin(dps, bin);
    }
  else if (const auto* proj_blk = dynamic_cast<const ProjDataInfoBlocksOnCylindricalNoArcCorr*>(proj_data_info_sptr.get()))
    {
      proj_blk->get_all_det_pos_pairs_for_bin(dps, bin);
    }
  else
    {
      error("BinNormalisationFromPETSIRD: ProjDataInfo is neither Cylindrical nor BlocksOnCylindrical");
    }
  if (dps.empty())
    {
      error("No detection position pairs found for bin.");
      return 1.f;
    }
  float total = 0.f;
  for (const auto& dp : dps)
    {
      float eff = petsird_info_sptr->get_detection_efficiency_for_bin(dp);
      if (m_use_dead_time)
        {
          const double start_time = get_exam_info_sptr()->get_time_frame_definitions().get_start_time();
          const double end_time = get_exam_info_sptr()->get_time_frame_definitions().get_end_time();
          eff *= get_dead_time_efficiency(dp.pos1(), start_time, end_time)
                 * get_dead_time_efficiency(dp.pos2(), start_time, end_time);
          // TODO at some point ModulePairs will need to be added here,
          // mMR doesn't use them, but Signa does
        }
      total += eff;
    }
  return total;
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
  auto header_sptr = std::make_shared<petsird::Header>();
  if (use_hdf5)
    petsird_data_sptr.reset(new petsird::hdf5::PETSIRDReader(filename));
  else
    petsird_data_sptr.reset(new petsird::binary::PETSIRDReader(filename));

  petsird_data_sptr->ReadHeader(*header_sptr);

  petsird_info_sptr = std::make_shared<PETSIRDInfo>(header_sptr);

  petsird::TimeBlock tb;
  while (petsird_data_sptr->ReadTimeBlocks(tb))
    {
      if (std::holds_alternative<petsird::DeadTimeTimeBlock>(tb))
        {
          const auto& db = std::get<petsird::DeadTimeTimeBlock>(tb);
          if (!db.alive_time_fractions.module_pair_alive_time_fractions.empty())
            error("BinNormalisationFromPETSIRD: module-pair alive-time fractions are not supported yet");
          dead_time_blocks.push_back(db);
        }
    }

  info(format("BinNormalisationFromPETSIRD: read {} dead time blocks", dead_time_blocks.size()));
}
float
BinNormalisationFromPETSIRD::get_dead_time_efficiency(const DetectionPosition<>& det_pos,
                                                      const double start_time,
                                                      const double end_time) const
{
  if (!m_use_dead_time)
    return 1.0F;
  if (dead_time_blocks.empty())
    error("BinNormalisationFromPETSIRD: dead time requested but the file contains no dead time blocks");

  // alive fractions are stored per detection bin (crystal), so look the crystal up in the STIR->PETSIRD map
  const auto& map = *petsird_info_sptr->get_stir_to_petsird_map();
  const auto it = map.find(det_pos);
  if (it == map.end())
    error("BinNormalisationFromPETSIRD: DetectionPosition not found in STIR->PETSIRD map");
  const auto crystal = petsird_helpers::make_detection_bin(petsird_info_sptr->get_petsird_scanner_info(), 0, it->second);

  // time-weighted average over the dead time blocks that overlap the frame
  double sum = 0;
  double total_time_ms = 0;
  for (const auto& b : dead_time_blocks)
    {
      const double overlap_ms = std::min<double>(b.time_interval.stop, end_time * 1000.)
                                - std::max<double>(b.time_interval.start, start_time * 1000.);
      if (overlap_ms > 0)
        {
          sum += b.alive_time_fractions.singles_alive_time_fractions[0][crystal] * overlap_ms;
          total_time_ms += overlap_ms;
        }
    }
  if (total_time_ms <= 0)
    error("BinNormalisationFromPETSIRD: no dead time blocks overlap the frame [%g, %g] s", start_time, end_time);
  return static_cast<float>(sum / total_time_ms);
}

END_NAMESPACE_STIR
