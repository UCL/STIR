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

START_NAMESPACE_STIR

const char* const BinNormalisationFromPETSIRD::registered_name = "From PETSIRD";

void
BinNormalisationFromPETSIRD::set_defaults()
{
  base_type::set_defaults();
  normalisation_filename = "";
  use_hdf5 = false;
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
  parser.add_key("use hdf5", &use_hdf5);
  parser.add_key("with_dead_time", &m_with_dead_time);
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
      if (m_with_dead_time)
        {
          const double start_time = get_exam_info_sptr()->get_time_frame_definitions().get_start_time();
          const double end_time = get_exam_info_sptr()->get_time_frame_definitions().get_end_time();
          static bool dbg = true;
          if (dbg)
            {
              dbg = false;
              std::cerr << "DEBUG PETSIRD DT frame: start=" << start_time << " end=" << end_time << "\n";
            }
          eff *= get_dead_time_efficiency(dp.pos1(), start_time, end_time)
                 * get_dead_time_efficiency(dp.pos2(), start_time, end_time);
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
  petsird::Header header;
  if (use_hdf5)
    petsird_data_sptr.reset(new petsird::hdf5::PETSIRDReader(filename));
  else
    petsird_data_sptr.reset(new petsird::binary::PETSIRDReader(filename));

  petsird_data_sptr->ReadHeader(header);

  petsird_info_sptr = std::make_shared<PETSIRDInfo>(header);

  // read pre-computed alive fractions from DeadTimeTimeBlock entries
  // alive fractions are stored per module
  const int num_modules = static_cast<int>(header.scanner.scanner_geometry.replicated_modules[0].transforms.size());
  num_buckets_dt = num_modules;

  int n_dt_blocks = 0;

  petsird::TimeBlock tb;
  while (petsird_data_sptr->ReadTimeBlocks(tb))
    {
      if (std::holds_alternative<petsird::DeadTimeTimeBlock>(tb))
        {
          const auto& db = std::get<petsird::DeadTimeTimeBlock>(tb);
          const auto& sat = db.alive_time_fractions.singles_alive_time_fractions;
          if (!sat.empty())
            {
              const auto& fracs = sat[0];
              DeadTimeEntry entry;
              entry.start_ms = db.time_interval.start;
              entry.stop_ms = db.time_interval.stop;
              if (static_cast<int>(fracs.size()) != num_modules)
                error("BinNormalisationFromPETSIRD: a dead time block has %d alive fractions, the scanner has %d modules",
                      static_cast<int>(fracs.size()), num_modules);
              entry.fractions.resize(num_modules);
              for (int i = 0; i < num_modules; ++i)
                entry.fractions[i] = static_cast<float>(fracs[i]);
              dead_time_entries.push_back(std::move(entry));
              ++n_dt_blocks;
            }
        }
    }

  std::cerr << "INFO BinNormalisationFromPETSIRD: read " << n_dt_blocks << " dead time blocks\n";
}
std::vector<float>
BinNormalisationFromPETSIRD::get_alive_fractions_for_frame(double start_time_s, double end_time_s) const
{
  const double start_ms = start_time_s * 1000.0;
  const double stop_ms = end_time_s * 1000.0;

  std::vector<float> fraction_sum(num_buckets_dt, 0.f);
  int count = 0;

  for (const auto& entry : dead_time_entries)
    {
      // include blocks that overlap with the frame window
      if (entry.stop_ms > start_ms && entry.start_ms < stop_ms)
        {
          for (int i = 0; i < num_buckets_dt; ++i)
            fraction_sum[i] += entry.fractions[i];
          ++count;
        }
    }

  if (count == 0)
    {
      error("BinNormalisationFromPETSIRD: no dead time blocks overlap the frame [%g, %g] s", start_time_s, end_time_s);
    }

  for (auto& f : fraction_sum)
    f /= count;

  return fraction_sum;
}
float
BinNormalisationFromPETSIRD::get_dead_time_efficiency(const DetectionPosition<>& det_pos,
                                                       const double start_time,
                                                       const double end_time) const
{
  if (!m_with_dead_time)
    return 1.0F;
  if (dead_time_entries.empty())
    error("BinNormalisationFromPETSIRD: dead time requested but the file contains no dead time blocks");

  // module of this detector, same convention as the convertor's detector IDs
  const Scanner& sc = *proj_data_info_sptr->get_scanner_ptr();
  const int ring = static_cast<int>(det_pos.axial_coord());
  const int tangential = static_cast<int>(det_pos.tangential_coord());
  const int module = ring / sc.get_num_axial_crystals_per_block()
                     + (tangential / sc.get_num_transaxial_crystals_per_block()) * sc.get_num_axial_blocks();

  const auto fracs = get_alive_fractions_for_frame(start_time, end_time);
  if (module < 0 || module >= static_cast<int>(fracs.size()))
    error("BinNormalisationFromPETSIRD: module %d (ring %d, tangential %d) outside the %d stored modules",
          module, ring, tangential, static_cast<int>(fracs.size()));
  return fracs[module];
}

END_NAMESPACE_STIR
