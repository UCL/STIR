//
//
/*
    Copyright (C) 2004, Hammersmith Imanet Ltd
    Copyright (C) 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisationPETFromComponents

  \author Kris Thielemans
*/

#include "stir/recon_buildblock/BinNormalisationPETFromComponents.h"
#include "stir/ProjDataInMemory.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Succeeded.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/numerics/divide.h"
#include "stir/format.h"

START_NAMESPACE_STIR

const char* const BinNormalisationPETFromComponents::registered_name = "PETComponents";

void
BinNormalisationPETFromComponents::set_defaults()
{
  base_type::set_defaults();
  this->_already_allocated = false;
  this->efficiencies.recycle();
  this->geo_data.recycle();
  this->block_data = BlockData3D();
}

BinNormalisationPETFromComponents::BinNormalisationPETFromComponents()
{
  set_defaults();
}

bool
BinNormalisationPETFromComponents::has_crystal_efficiencies() const
{
  return this->efficiencies.size() > 0;
}

bool
BinNormalisationPETFromComponents::has_geometric_factors() const
{
  return this->geo_data.size() > 0;
}

bool
BinNormalisationPETFromComponents::has_block_factors() const
{
  return this->block_data.get_num_detectors_per_ring() > 0;
}

Succeeded
BinNormalisationPETFromComponents::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr,
                                          const shared_ptr<const ProjDataInfo>& check_proj_data_info_sptr)
{
  if (!_already_allocated)
    error("BinNormalisationPETFromComponents: set_up called without allocation");

  base_type::set_up(exam_info_sptr, check_proj_data_info_sptr);

  if (*this->proj_data_info_sptr != *check_proj_data_info_sptr)
    return Succeeded::no;
  {
    const ProjDataInfo& norm_proj = *this->proj_data_info_sptr;
    const ProjDataInfo& proj = *check_proj_data_info_sptr;
    bool ok = (norm_proj >= proj) && (norm_proj.get_min_tangential_pos_num() == proj.get_min_tangential_pos_num())
              && (norm_proj.get_max_tangential_pos_num() == proj.get_max_tangential_pos_num());

    for (int segment_num = proj.get_min_segment_num(); ok && segment_num <= proj.get_max_segment_num(); ++segment_num)
      {
        ok = norm_proj.get_min_axial_pos_num(segment_num) == proj.get_min_axial_pos_num(segment_num)
             && norm_proj.get_max_axial_pos_num(segment_num) == proj.get_max_axial_pos_num(segment_num);
      }
    if (!ok)
      {
        warning(format("BinNormalisationPETFromComponents: incompatible projection data:\nNorm projdata "
                       "info:\n{}\nEmission projdata info:\n{}\n--- (end of incompatible projection data info)---\n",
                       norm_proj.parameter_info(),
                       proj.parameter_info()));
        return Succeeded::no;
      }
  }

  this->_is_trivial
      = (!has_crystal_efficiencies()
         || (fabs(efficiencies.find_min() - 1) <= .0001 && fabs(efficiencies.find_max() - 1) <= .0001))
        && (!has_geometric_factors() || (fabs(geo_data.find_min() - 1) <= .0001 && fabs(geo_data.find_max() - 1) <= .0001))
        && (!has_block_factors() || (fabs(block_data.find_min() - 1) <= .0001 && fabs(block_data.find_max() - 1) <= .0001));

  this->create_proj_data();
  return Succeeded::yes;
}

bool
BinNormalisationPETFromComponents::is_trivial() const
{
  if (!this->_already_set_up)
    error("BinNormalisationPETFromComponents: is_trivial called without set_up");
  return this->_is_trivial;
}

void
BinNormalisationPETFromComponents::allocate(
    shared_ptr<const ProjDataInfo> pdi_sptr, bool do_eff, bool do_geo, bool do_block, bool do_symmetry_per_block)
{
  this->proj_data_info_sptr = pdi_sptr;

  const int num_transaxial_blocks = proj_data_info_sptr->get_scanner_sptr()->get_num_transaxial_blocks();
  const int num_axial_blocks = proj_data_info_sptr->get_scanner_sptr()->get_num_axial_blocks();
  const int virtual_axial_crystals = proj_data_info_sptr->get_scanner_sptr()->get_num_virtual_axial_crystals_per_block();
  const int virtual_transaxial_crystals
      = proj_data_info_sptr->get_scanner_sptr()->get_num_virtual_transaxial_crystals_per_block();
  const int num_physical_rings
      = proj_data_info_sptr->get_scanner_sptr()->get_num_rings() - (num_axial_blocks - 1) * virtual_axial_crystals;
  const int num_physical_detectors_per_ring = proj_data_info_sptr->get_scanner_sptr()->get_num_detectors_per_ring()
                                              - num_transaxial_blocks * virtual_transaxial_crystals;
  const int num_transaxial_buckets = proj_data_info_sptr->get_scanner_sptr()->get_num_transaxial_buckets();
  const int num_axial_buckets = proj_data_info_sptr->get_scanner_sptr()->get_num_axial_buckets();
  const int num_transaxial_blocks_per_bucket = proj_data_info_sptr->get_scanner_sptr()->get_num_transaxial_blocks_per_bucket();
  const int num_axial_blocks_per_bucket = proj_data_info_sptr->get_scanner_sptr()->get_num_axial_blocks_per_bucket();

  int num_physical_transaxial_crystals_per_basic_unit
      = proj_data_info_sptr->get_scanner_sptr()->get_num_transaxial_crystals_per_block() - virtual_transaxial_crystals;
  int num_physical_axial_crystals_per_basic_unit
      = proj_data_info_sptr->get_scanner_sptr()->get_num_axial_crystals_per_block() - virtual_axial_crystals;
  // If there are multiple buckets, we increase the symmetry size to a bucket. Otherwise, we use a block.
  if (do_symmetry_per_block == false)
    {
      if (num_transaxial_buckets > 1)
        {
          num_physical_transaxial_crystals_per_basic_unit *= num_transaxial_blocks_per_bucket;
        }
      if (num_axial_buckets > 1)
        {
          num_physical_axial_crystals_per_basic_unit *= num_axial_blocks_per_bucket;
        }
    }

  if (do_geo)
    this->geo_data = GeoData3D(num_physical_axial_crystals_per_basic_unit,
                               num_physical_transaxial_crystals_per_basic_unit / 2,
                               num_physical_rings,
                               num_physical_detectors_per_ring);
  else
    this->geo_data.recycle();

  if (do_block)
    this->block_data = BlockData3D(num_axial_blocks, num_transaxial_blocks, num_axial_blocks - 1, num_transaxial_blocks - 1);
  else
    this->block_data = BlockData3D();
  ;

  if (do_eff)
    this->efficiencies.resize(IndexRange2D(num_physical_rings, num_physical_detectors_per_ring));
  else
    this->efficiencies.recycle();

  this->_already_allocated = true;
}

void
BinNormalisationPETFromComponents::create_proj_data()
{
  if (!_already_allocated)
    error("BinNormalisationPETFromComponents: internal error: create_proj_data called without allocation");
  if (!_already_set_up)
    error("BinNormalisationPETFromComponents: internal error: create_proj_data called without set_up");

  FanProjData fan_data;

  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  this->invnorm_proj_data_sptr
      = std::make_shared<ProjDataInMemory>(exam_info_sptr, this->proj_data_info_sptr, /* do not initialise */ false);
  this->invnorm_proj_data_sptr->fill(1.F);
  make_fan_data_remove_gaps(fan_data, *this->invnorm_proj_data_sptr);

  // multiply fan data with resp factors
  if (this->has_block_factors())
    apply_block_norm(fan_data, this->block_data, true);
  if (this->has_crystal_efficiencies())
    apply_efficiencies(fan_data, this->efficiencies, true);
  if (this->has_geometric_factors())
    apply_geo_norm(fan_data, this->geo_data, true);

  set_fan_data_add_gaps(*invnorm_proj_data_sptr, fan_data);
}

// as in BinNormalisationFromProjData
void
BinNormalisationPETFromComponents::apply(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  const auto vs_num = viewgrams.get_basic_view_segment_num();
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());

  // divide, but need to handle 0/0
  const auto invnorm_relv = invnorm_proj_data_sptr->get_related_viewgrams(vs_num, symmetries_sptr, false);
  auto v_iter = viewgrams.begin();
  auto invnorm_v_iter = invnorm_relv.begin();
  while (v_iter != viewgrams.end())
    {
      divide(v_iter->begin_all(), v_iter->end_all(), invnorm_v_iter->begin_all(), 0.F);
      ++v_iter;
      ++invnorm_v_iter;
    }
}

void
BinNormalisationPETFromComponents::undo(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  const auto vs_num = viewgrams.get_basic_view_segment_num();
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(viewgrams.get_symmetries_ptr()->clone());
  viewgrams *= invnorm_proj_data_sptr->get_related_viewgrams(vs_num, symmetries_sptr, false);
}

float
BinNormalisationPETFromComponents::get_bin_efficiency(const Bin& bin) const
{
  // need a copy at the moment
  Bin copy(bin);
  return this->invnorm_proj_data_sptr->get_bin_value(copy);
}

#if 0
shared_ptr<ProjData>
BinNormalisationPETFromComponents::get_norm_proj_data_sptr() const
{
  return this->norm_proj_data_ptr;
}
#endif

END_NAMESPACE_STIR
