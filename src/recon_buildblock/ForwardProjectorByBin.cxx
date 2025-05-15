//
//
/*!

  \file
  \ingroup projection

  \brief non-inline implementations for stir::ForwardProjectorByBin

  \author Kris Thielemans
  \author PARAPET project
  \author Richard Brown


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013 Kris Thielemans
    Copyright (C) 2015, 2018-2019, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/Succeeded.h"
#include "stir/info.h"
#include "stir/error.h"
#include "stir/warning.h"
#include "stir/format.h"
#include "stir/DataProcessor.h"
#include "stir/is_null_ptr.h"
#include <iostream>

START_NAMESPACE_STIR

ForwardProjectorByBin::ForwardProjectorByBin()
    : _already_set_up(false)
{
  set_defaults();
}

ForwardProjectorByBin::~ForwardProjectorByBin()
{}

void
ForwardProjectorByBin::set_defaults()
{
  _pre_data_processor_sptr.reset();
}

void
ForwardProjectorByBin::initialise_keymap()
{
  parser.add_start_key("Forward Projector Parameters");
  parser.add_stop_key("End Forward Projector Parameters");
  parser.add_parsing_key("pre data processor", &_pre_data_processor_sptr);
}

void
ForwardProjectorByBin::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                              const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  _already_set_up = true;
  _proj_data_info_sptr = proj_data_info_sptr->create_shared_clone();
  _density_sptr.reset(density_info_sptr->clone());
}

void
ForwardProjectorByBin::check(const ProjDataInfo& proj_data_info) const
{
  if (!this->_already_set_up)
    error("ForwardProjectorByBin method called without calling set_up first.");
  if (!(*this->_proj_data_info_sptr >= proj_data_info))
    error(
        format("ForwardProjectorByBin set-up with different geometry for projection data.\nSet_up was with\n{}\nCalled with\n{}",
               this->_proj_data_info_sptr->parameter_info(),
               proj_data_info.parameter_info()));
}

void
ForwardProjectorByBin::check(const ProjDataInfo& proj_data_info, const DiscretisedDensity<3, float>& density_info) const
{
  this->check(proj_data_info);
  if (!this->_density_sptr->has_same_characteristics(density_info))
    error("ForwardProjectorByBin set-up with different geometry for density or volume data.");
}

void
ForwardProjectorByBin::forward_project(
    ProjData& proj_data, const DiscretisedDensity<3, float>& image, int subset_num, int num_subsets, bool zero)
{
  set_input(image);
  forward_project(proj_data, subset_num, num_subsets, zero);
}
#ifdef STIR_PROJECTORS_AS_V3
void
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams, const DiscretisedDensity<3, float>& image)
{
  forward_project(viewgrams,
                  image,
                  viewgrams.get_min_axial_pos_num(),
                  viewgrams.get_max_axial_pos_num(),
                  viewgrams.get_min_tangential_pos_num(),
                  viewgrams.get_max_tangential_pos_num());
}

void
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams,
                                       const DiscretisedDensity<3, float>& image,
                                       const int min_axial_pos_num,
                                       const int max_axial_pos_num)
{
  forward_project(viewgrams,
                  image,
                  min_axial_pos_num,
                  max_axial_pos_num,
                  viewgrams.get_min_tangential_pos_num(),
                  viewgrams.get_max_tangential_pos_num());
}

void
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams,
                                       const DiscretisedDensity<3, float>& density,
                                       const int min_axial_pos_num,
                                       const int max_axial_pos_num,
                                       const int min_tangential_pos_num,
                                       const int max_tangential_pos_num)
{
  if (viewgrams.get_num_viewgrams() == 0)
    return;
  check(*viewgrams.get_proj_data_info_sptr(), density);

  // first check symmetries
  {
    const ViewSegmentNumbers basic_vs = viewgrams.get_basic_view_segment_num();

    if (get_symmetries_used()->num_related_view_segment_numbers(basic_vs) != viewgrams.get_num_viewgrams())
      error("ForwardProjectByBin: forward_project called with incorrect related_viewgrams (wrong number). Problem with "
            "symmetries!");

    for (RelatedViewgrams<float>::const_iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
      {
        ViewSegmentNumbers vs(iter->get_view_num(), iter->get_segment_num());
        get_symmetries_used()->find_basic_view_segment_numbers(vs);
        // TODOTOF find_basic_view_segment_numbers doesn't fill in timing_pos_num
        vs.timing_pos_num() = basic_vs.timing_pos_num();
        if (vs != basic_vs)
          error("ForwardProjectByBin: forward_project called with incorrect related_viewgrams. Problem with symmetries!\n");
      }
  }
  actual_forward_project(
      viewgrams, density, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
  stop_timers();
}
#endif
// -------------------------------------------------------------------------------------------------------------------- //
// The following are repetition of above, where the DiscretisedDensity has already been set with set_input()
// -------------------------------------------------------------------------------------------------------------------- //
void
ForwardProjectorByBin::forward_project(ProjData& proj_data, int subset_num, int num_subsets, bool zero)
{
  if (!_density_sptr)
    error("You need to call set_input() forward_project()");

  if (_density_sptr->get_exam_info().imaging_modality.is_unknown() || proj_data.get_exam_info().imaging_modality.is_unknown())
    warning("forward_project. Imaging modality unknown for either the image or the projection data or both.\n"
            "Going ahead anyway.");
  else if (_density_sptr->get_exam_info().imaging_modality != proj_data.get_exam_info().imaging_modality)
    error("forward_project: Imaging modality should be the same for the image and the projection data");
  if (subset_num < 0)
    error(format("forward_project: wrong subset number {}", subset_num));
  if (subset_num > num_subsets - 1)
    error(
        format("forward_project: wrong subset number {} (must be less than the number of subsets {})", subset_num, num_subsets));
  if (zero && num_subsets > 1)
    proj_data.fill(0.0);
  // this->set_up(proj_data_ptr->get_proj_data_info_sptr()->clone(),
  //			     image_sptr);

  check(*proj_data.get_proj_data_info_sptr(), *_density_sptr);
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(this->get_symmetries_used()->clone());

  const std::vector<ViewSegmentNumbers> vs_nums_to_process
      = detail::find_basic_vs_nums_in_subset(*proj_data.get_proj_data_info_sptr(),
                                             *symmetries_sptr,
                                             proj_data.get_min_segment_num(),
                                             proj_data.get_max_segment_num(),
                                             subset_num,
                                             num_subsets);
#ifdef STIR_OPENMP
#  if _OPENMP < 201107
#    pragma omp parallel for shared(proj_data, symmetries_sptr) schedule(dynamic)
#  else
// OpenMP loop over both vs_nums_to_process and tof_pos_num
#    pragma omp parallel for shared(proj_data, symmetries_sptr) schedule(dynamic) collapse(2)
#  endif
#endif
  // note: older versions of openmp need an int as loop
  for (int i = 0; i < static_cast<int>(vs_nums_to_process.size()); ++i)
    {
      for (int k = proj_data.get_proj_data_info_sptr()->get_min_tof_pos_num();
           k <= proj_data.get_proj_data_info_sptr()->get_max_tof_pos_num();
           ++k)
        {
          const ViewSegmentNumbers vs = vs_nums_to_process[i];
          if (proj_data.get_proj_data_info_sptr()->is_tof_data())
            info(format("Processing view {} of segment {} of TOF bin {}", vs.view_num(), vs.segment_num(), k), 3);
          else
            info(format("Processing view {} of segment {}", vs.view_num(), vs.segment_num()), 3);
          RelatedViewgrams<float> viewgrams = proj_data.get_empty_related_viewgrams(vs, symmetries_sptr, false, k);
          forward_project(viewgrams);
#ifdef STIR_OPENMP
#  pragma omp critical(FORWARDPROJ_SETVIEWGRAMS)
#endif
          {
            if (!(proj_data.set_related_viewgrams(viewgrams) == Succeeded::yes))
              error("Error set_related_viewgrams in forward projecting");
          }
        }
    }
}

void
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams)
{
  forward_project(viewgrams,
                  viewgrams.get_min_axial_pos_num(),
                  viewgrams.get_max_axial_pos_num(),
                  viewgrams.get_min_tangential_pos_num(),
                  viewgrams.get_max_tangential_pos_num());
}

void
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams,
                                       const int min_axial_pos_num,
                                       const int max_axial_pos_num)
{
  forward_project(viewgrams,
                  min_axial_pos_num,
                  max_axial_pos_num,
                  viewgrams.get_min_tangential_pos_num(),
                  viewgrams.get_max_tangential_pos_num());
}

void
ForwardProjectorByBin::forward_project(RelatedViewgrams<float>& viewgrams,
                                       const int min_axial_pos_num,
                                       const int max_axial_pos_num,
                                       const int min_tangential_pos_num,
                                       const int max_tangential_pos_num)
{
  if (viewgrams.get_num_viewgrams() == 0)
    return;
  if (!_density_sptr)
    error("You need to call set_input() forward_project()");

  check(*viewgrams.get_proj_data_info_sptr());

  // first check symmetries
  {
    const ViewSegmentNumbers basic_vs = viewgrams.get_basic_view_segment_num();

    if (get_symmetries_used()->num_related_view_segment_numbers(basic_vs) != viewgrams.get_num_viewgrams())
      error("ForwardProjectByBin: forward_project called with incorrect related_viewgrams. Problem with symmetries!\n");

    for (RelatedViewgrams<float>::const_iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
      {
        ViewSegmentNumbers vs(iter->get_view_num(), iter->get_segment_num());
        get_symmetries_used()->find_basic_view_segment_numbers(vs);
        // TODOTOF find_basic_view_segment_numbers doesn't fill in timing_pos_num
        vs.timing_pos_num() = basic_vs.timing_pos_num();
        if (vs != basic_vs)
          error("ForwardProjectByBin: forward_project called with incorrect related_viewgrams. Problem with symmetries!\n");
      }
  }
  actual_forward_project(viewgrams, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
}

void
ForwardProjectorByBin::actual_forward_project(
    RelatedViewgrams<float>&, const DiscretisedDensity<3, float>&, const int, const int, const int, const int)
{
  error("ForwardProjectorByBin::actual_forward_project() This is deprecated and should not be used.");
}

void
ForwardProjectorByBin::actual_forward_project(RelatedViewgrams<float>& viewgrams,
                                              const int min_axial_pos_num,
                                              const int max_axial_pos_num,
                                              const int min_tangential_pos_num,
                                              const int max_tangential_pos_num)
{
  actual_forward_project(
      viewgrams, *_density_sptr, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
}

void
ForwardProjectorByBin::set_input(const DiscretisedDensity<3, float>& density)
{
  _density_sptr.reset(density.clone());

  // If a pre-forward-projection data processor has been set, apply it.
  if (!is_null_ptr(_pre_data_processor_sptr))
    {
      Succeeded success = _pre_data_processor_sptr->apply(*_density_sptr);
      if (success != Succeeded::yes)
        throw std::runtime_error("ForwardProjectorByBin::set_input(). Pre-forward-projection data processor failed.");
    }
}

void
ForwardProjectorByBin::set_pre_data_processor(shared_ptr<DataProcessor<DiscretisedDensity<3, float>>> pre_data_processor_sptr)
{
  _pre_data_processor_sptr = pre_data_processor_sptr;
}

END_NAMESPACE_STIR
