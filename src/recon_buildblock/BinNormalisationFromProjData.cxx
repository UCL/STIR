//
//
/*
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2023, University College London
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
#include <boost/format.hpp>

START_NAMESPACE_STIR

const char * const 
BinNormalisationFromProjData::registered_name = "From ProjData"; 


void 
BinNormalisationFromProjData::set_defaults()
{
  base_type::set_defaults();
  normalisation_projdata_filename = "";
}

void 
BinNormalisationFromProjData::
initialise_keymap()
{
  base_type::initialise_keymap();
  parser.add_start_key("Bin Normalisation From ProjData");
  parser.add_key("normalisation_projdata_filename", &normalisation_projdata_filename);
  parser.add_stop_key("End Bin Normalisation From ProjData");
}

bool 
BinNormalisationFromProjData::
post_processing()
{
  if (base_type::post_processing())
    return true;
  norm_proj_data_ptr = ProjData::read_from_file(normalisation_projdata_filename);
  return false;
}

BinNormalisationFromProjData::
BinNormalisationFromProjData()
{
  set_defaults();
}

BinNormalisationFromProjData::
BinNormalisationFromProjData(const std::string& filename)
    : norm_proj_data_ptr(ProjData::read_from_file(filename))
  {}

BinNormalisationFromProjData::
BinNormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr)
    : norm_proj_data_ptr(norm_proj_data_ptr)
  {}

Succeeded 
BinNormalisationFromProjData::
set_up(const shared_ptr<const ExamInfo>& exam_info_sptr, const shared_ptr<const ProjDataInfo>& proj_data_info_ptr)
{
  base_type::set_up(exam_info_sptr, proj_data_info_ptr);

  if (*(norm_proj_data_ptr->get_proj_data_info_sptr()) == *proj_data_info_ptr)
    return Succeeded::yes;
  else
  {
    const ProjDataInfo& norm_proj = *(norm_proj_data_ptr->get_proj_data_info_sptr());
    const ProjDataInfo& proj = *proj_data_info_ptr;
    bool ok = 
      (norm_proj >= proj) &&
      (norm_proj.get_min_tangential_pos_num() ==proj.get_min_tangential_pos_num())&&
      (norm_proj.get_max_tangential_pos_num() ==proj.get_max_tangential_pos_num());

    if (ok)
      return Succeeded::yes;
    else
      {
	warning(boost::format("BinNormalisationFromProjData: incompatible projection data:\nNorm projdata info:\n%s\nEmission projdata info:\n%s\n--- (end of incompatible projection data info)---\n")
		% norm_proj.parameter_info()
		% proj.parameter_info());
	return Succeeded::no;
      }
  }
}

bool 
BinNormalisationFromProjData::
is_trivial() const
{
  return false;
}

void 
BinNormalisationFromProjData::apply(RelatedViewgrams<float>& viewgrams) const 
  {
    this->check(*viewgrams.get_proj_data_info_sptr());
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
	const int timing_pos_num = norm_proj_data_ptr->get_proj_data_info_sptr()->is_tof_data() ? viewgrams.get_basic_timing_pos_num() : 0;
    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
    viewgrams *= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_sptr, false,timing_pos_num);
  }

void 
BinNormalisationFromProjData::
undo(RelatedViewgrams<float>& viewgrams) const 
  {
    this->check(*viewgrams.get_proj_data_info_sptr());
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
	const int timing_pos_num = norm_proj_data_ptr->get_proj_data_info_sptr()->is_tof_data() ? viewgrams.get_basic_timing_pos_num() : 0;
    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
    viewgrams /= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_sptr, false, timing_pos_num);

  }

float 
BinNormalisationFromProjData::get_bin_efficiency(const Bin& bin) const
{
  //TODO
  error("BinNormalisationFromProjData::get_bin_efficiency is not implemented");
  return 1;

}

shared_ptr<ProjData>
BinNormalisationFromProjData::get_norm_proj_data_sptr() const
{
  return this->norm_proj_data_ptr;
}
 
END_NAMESPACE_STIR
