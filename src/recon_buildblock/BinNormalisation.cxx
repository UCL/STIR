//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2014, 2018 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisation

  \author Kris Thielemans
*/

#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/recon_buildblock/TrivialDataSymmetriesForBins.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Bin.h"
#include "stir/ProjData.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

BinNormalisation::BinNormalisation()
    : _already_set_up(false)
{}

void
BinNormalisation::set_defaults()
{
  this->_already_set_up = false;
}

BinNormalisation::~BinNormalisation()
{}

void
BinNormalisation::set_exam_info_sptr(const shared_ptr<const ExamInfo> _exam_info_sptr)
{
  this->exam_info_sptr = _exam_info_sptr;
}

shared_ptr<const ExamInfo>
BinNormalisation::get_exam_info_sptr() const
{
  return this->exam_info_sptr;
}

Succeeded
BinNormalisation::set_up(const shared_ptr<const ExamInfo>& exam_info_sptr_v,
                         const shared_ptr<const ProjDataInfo>& proj_data_info_sptr_v)
{
  _already_set_up = true;
  this->proj_data_info_sptr = proj_data_info_sptr_v;
  this->exam_info_sptr = exam_info_sptr_v;
  return Succeeded::yes;
}

void
BinNormalisation::check(const ProjDataInfo& proj_data_info) const
{
  if (!this->_already_set_up)
    error("BinNormalisation method called without calling set_up first.");
  if (!(*this->proj_data_info_sptr >= proj_data_info))
    error(format("BinNormalisation set-up with different geometry for projection data.\nSet_up was with\n{}\nCalled with\n{}",
                 this->proj_data_info_sptr->parameter_info(),
                 proj_data_info.parameter_info()));
}

void
BinNormalisation::check(const ExamInfo& exam_info) const
{
  if (!(*this->exam_info_sptr == exam_info))
    error(format("BinNormalisation set-up with different ExamInfo.\n Set_up was with\n{}\nCalled with\n{}",
                 this->exam_info_sptr->parameter_info(),
                 exam_info.parameter_info()));
}

// TODO remove duplication between apply and undo by just having 1 functino that does the loops

void
BinNormalisation::apply(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
    {
      Bin bin(iter->get_segment_num(), iter->get_view_num(), 0, 0, iter->get_timing_pos_num());
      for (bin.axial_pos_num() = iter->get_min_axial_pos_num(); bin.axial_pos_num() <= iter->get_max_axial_pos_num();
           ++bin.axial_pos_num())
        for (bin.tangential_pos_num() = iter->get_min_tangential_pos_num();
             bin.tangential_pos_num() <= iter->get_max_tangential_pos_num();
             ++bin.tangential_pos_num())
          (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= std::max(1.E-20F, get_bin_efficiency(bin));
    }
}

void
BinNormalisation::undo(RelatedViewgrams<float>& viewgrams) const
{
  this->check(*viewgrams.get_proj_data_info_sptr());
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
    {
      Bin bin(iter->get_segment_num(), iter->get_view_num(), 0, 0, iter->get_timing_pos_num());
      for (bin.axial_pos_num() = iter->get_min_axial_pos_num(); bin.axial_pos_num() <= iter->get_max_axial_pos_num();
           ++bin.axial_pos_num())
        for (bin.tangential_pos_num() = iter->get_min_tangential_pos_num();
             bin.tangential_pos_num() <= iter->get_max_tangential_pos_num();
             ++bin.tangential_pos_num())
          (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= this->get_bin_efficiency(bin);
    }
}

void
BinNormalisation::apply(ProjData& proj_data, shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr) const
{
  this->check(*proj_data.get_proj_data_info_sptr());
  this->check(proj_data.get_exam_info());
  if (is_null_ptr(symmetries_sptr))
    symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data.get_proj_data_info_sptr()->create_shared_clone()));

  const std::vector<ViewSegmentNumbers> vs_nums_to_process
      = detail::find_basic_vs_nums_in_subset(*proj_data.get_proj_data_info_sptr(),
                                             *symmetries_sptr,
                                             proj_data.get_min_segment_num(),
                                             proj_data.get_max_segment_num(),
                                             0,
                                             1 /*subset_num, num_subsets*/);

#ifdef STIR_OPENMP
#  pragma omp parallel for shared(proj_data, symmetries_sptr) schedule(dynamic)
#endif
  // note: older versions of openmp need an int as loop
  for (int i = 0; i < static_cast<int>(vs_nums_to_process.size()); ++i)
    {
      const ViewSegmentNumbers vs = vs_nums_to_process[i];

      for (int k = proj_data.get_proj_data_info_sptr()->get_min_tof_pos_num();
           k <= proj_data.get_proj_data_info_sptr()->get_max_tof_pos_num();
           ++k)
        {

          RelatedViewgrams<float> viewgrams;
#ifdef STIR_OPENMP
          // reading/writing to streams is not safe in multi-threaded code
          // so protect with a critical section
          // note that the name of the section has to be same for the get/set
          // function as they're reading from/writing to the same stream
#  pragma omp critical(BINNORMALISATION_APPLY__VIEWGRAMS)
#endif
          {
            viewgrams = proj_data.get_related_viewgrams(vs, symmetries_sptr, false, k);
          }

          this->apply(viewgrams);

#ifdef STIR_OPENMP
#  pragma omp critical(BINNORMALISATION_APPLY__VIEWGRAMS)
#endif
          {
            proj_data.set_related_viewgrams(viewgrams);
          }
        }
    }
}

void
BinNormalisation::undo(ProjData& proj_data, shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr) const
{
  this->check(*proj_data.get_proj_data_info_sptr());
  this->check(proj_data.get_exam_info());
  if (is_null_ptr(symmetries_sptr))
    symmetries_sptr.reset(new TrivialDataSymmetriesForBins(proj_data.get_proj_data_info_sptr()->create_shared_clone()));

  const std::vector<ViewSegmentNumbers> vs_nums_to_process
      = detail::find_basic_vs_nums_in_subset(*proj_data.get_proj_data_info_sptr(),
                                             *symmetries_sptr,
                                             proj_data.get_min_segment_num(),
                                             proj_data.get_max_segment_num(),
                                             0,
                                             1 /*subset_num, num_subsets*/);

#ifdef STIR_OPENMP
#  pragma omp parallel for shared(proj_data, symmetries_sptr) schedule(dynamic)
#endif
  // note: older versions of openmp need an int as loop
  for (int i = 0; i < static_cast<int>(vs_nums_to_process.size()); ++i)
    {
      const ViewSegmentNumbers vs = vs_nums_to_process[i];

      for (int k = proj_data.get_proj_data_info_sptr()->get_min_tof_pos_num();
           k <= proj_data.get_proj_data_info_sptr()->get_max_tof_pos_num();
           ++k)
        {
          RelatedViewgrams<float> viewgrams;
#ifdef STIR_OPENMP
#  pragma omp critical(BINNORMALISATION_UNDO__VIEWGRAMS)
#endif
          {
            viewgrams = proj_data.get_related_viewgrams(vs, symmetries_sptr, false, k);
          }

          this->undo(viewgrams);

#ifdef STIR_OPENMP
#  pragma omp critical(BINNORMALISATION_UNDO__VIEWGRAMS)
#endif
          {
            proj_data.set_related_viewgrams(viewgrams);
          }
        }
    }
}

END_NAMESPACE_STIR
