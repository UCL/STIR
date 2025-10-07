//
//
/*
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2023, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationFromProjData

  \author Kris Thielemans
*/

#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Succeeded.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

const char* const BinNormalisationFromProjData::registered_name = "From ProjData";

void
BinNormalisationFromProjData::set_defaults()
{
  base_type::set_defaults();
  normalisation_projdata_filename = "";
}

void
BinNormalisationFromProjData::initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Bin Normalisation From ProjData");
  parser.add_key("normalisation_projdata_filename", &normalisation_projdata_filename);
  parser.add_stop_key("End Bin Normalisation From ProjData");
}

bool
BinNormalisationFromProjData::post_processing()
{
  if (base_type::post_processing())
    return true;
  norm_proj_data_ptr = ProjData::read_from_file(normalisation_projdata_filename);
  return false;
}

BinNormalisationFromProjData::BinNormalisationFromProjData()
{
  set_defaults();
}

BinNormalisationFromProjData::BinNormalisationFromProjData(const std::string& filename)
    : norm_proj_data_ptr(ProjData::read_from_file(filename))
{}

BinNormalisationFromProjData::BinNormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr)
    : norm_proj_data_ptr(norm_proj_data_ptr)
{}

bool
BinNormalisationFromProjData::is_TOF_only_norm() const
{
  if (!this->get_norm_proj_data_sptr())
    error("BinNormalisationFromProjData: projection data not set.");
  return this->get_norm_proj_data_sptr()->get_num_tof_poss() > 1;
}

Succeeded
BinNormalisationFromProjData::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,
                                     const shared_ptr<const ProjDataInfo>& proj_data_info_sptr)
{
  if (!base_type::set_up(exam_info_sptr, proj_data_info_sptr).succeeded())
    return Succeeded::no;

  const auto& norm_proj = *(norm_proj_data_ptr->get_proj_data_info_sptr());
  // complication: if the emission data is TOF but the norm is not, `apply()` et al. multiply all
  // TOF bins with the non-TOF norm. So, we need to allow for that.
  auto proj_to_check_sptr = proj_data_info_sptr;
  if (!norm_proj.is_tof_data() && proj_data_info_sptr->is_tof_data())
    proj_to_check_sptr = proj_data_info_sptr->create_non_tof_clone();
  const auto& proj = *proj_to_check_sptr;

  if (norm_proj == proj)
    return Succeeded::yes;
  else
    {
      // Check if the emission data is "smaller" than the norm data (e.g. fewer segments)
      // We will require equal tangential_pos ranges as `apply()` currently needs that.
      bool ok = (norm_proj >= proj) && (norm_proj.get_min_tangential_pos_num() == proj.get_min_tangential_pos_num())
                && (norm_proj.get_max_tangential_pos_num() == proj.get_max_tangential_pos_num());

      for (int segment_num = proj.get_min_segment_num(); ok && segment_num <= proj.get_max_segment_num(); ++segment_num)
        {
          ok = norm_proj.get_min_axial_pos_num(segment_num) == proj.get_min_axial_pos_num(segment_num)
               && norm_proj.get_max_axial_pos_num(segment_num) == proj.get_max_axial_pos_num(segment_num);
        }
      if (ok)
        return Succeeded::yes;
      else
        {
          warning(
              format("BinNormalisationFromProjData: incompatible projection data:\nNorm projdata info:\n{}\nEmission projdata "
                     "info (made non-TOF if norm is non-TOF):\n{}\n--- (end of incompatible projection data info)---\n",
                     norm_proj.parameter_info(),
                     proj.parameter_info()));
          return Succeeded::no;
        }
    }
}

bool
BinNormalisationFromProjData::is_trivial() const
{
  return false;
}

void
BinNormalisationFromProjData::apply(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  const ViewSegmentNumbers vs_num = viewgrams.get_basic_view_segment_num();
  const int timing_pos_num
      = norm_proj_data_ptr->get_proj_data_info_sptr()->is_tof_data() ? viewgrams.get_basic_timing_pos_num() : 0;
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
  viewgrams *= norm_proj_data_ptr->get_related_viewgrams(vs_num, symmetries_sptr, false, timing_pos_num);
}

void
BinNormalisationFromProjData::undo(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  const ViewSegmentNumbers vs_num = viewgrams.get_basic_view_segment_num();
  const int timing_pos_num
      = norm_proj_data_ptr->get_proj_data_info_sptr()->is_tof_data() ? viewgrams.get_basic_timing_pos_num() : 0;
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
  viewgrams /= norm_proj_data_ptr->get_related_viewgrams(vs_num, symmetries_sptr, false, timing_pos_num);
}

float
BinNormalisationFromProjData::get_bin_efficiency(const Bin& bin) const
{
  // TODO
  error("BinNormalisationFromProjData::get_bin_efficiency is not implemented");
  return 1;
}

shared_ptr<ProjData>
BinNormalisationFromProjData::get_norm_proj_data_sptr() const
{
  return this->norm_proj_data_ptr;
}

END_NAMESPACE_STIR
