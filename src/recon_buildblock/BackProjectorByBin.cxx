//
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::BackProjectorByBin

  \author Kris Thielemans
  \author PARAPET project
  \author Richard Brown

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2015, 2018-2019, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/recon_buildblock/find_basic_vs_nums_in_subsets.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ProjData.h"
#include "stir/DiscretisedDensity.h"
#include "stir/info.h"
#include "stir/error.h"
#include "stir/format.h"
#include "stir/is_null_ptr.h"
#include "stir/DataProcessor.h"
#include <vector>
#ifdef STIR_OPENMP
#  include "stir/is_null_ptr.h"
#  include "stir/DiscretisedDensity.h"
#  include <omp.h>
#endif

START_NAMESPACE_STIR

BackProjectorByBin::BackProjectorByBin()
    : _already_set_up(false)
{
  set_defaults();
}

BackProjectorByBin::~BackProjectorByBin()
{}

void
BackProjectorByBin::set_defaults()
{
  _post_data_processor_sptr.reset();
}

void
BackProjectorByBin::initialise_keymap()
{
  parser.add_start_key("Back Projector Parameters");
  parser.add_stop_key("End Back Projector Parameters");
  parser.add_parsing_key("post data processor", &_post_data_processor_sptr);
}

void
BackProjectorByBin::set_up(const shared_ptr<const ProjDataInfo>& proj_data_info_sptr,
                           const shared_ptr<const DiscretisedDensity<3, float>>& density_info_sptr)
{
  _already_set_up = true;
  _proj_data_info_sptr = proj_data_info_sptr->create_shared_clone();
  _density_sptr.reset(density_info_sptr->clone());

#ifdef STIR_OPENMP
#  pragma omp parallel
  {
#  pragma omp single
    _local_output_image_sptrs.resize(omp_get_num_threads(), shared_ptr<DiscretisedDensity<3, float>>());
  }
  for (int i = 0; i < static_cast<int>(_local_output_image_sptrs.size()); ++i)
    if (!is_null_ptr(_local_output_image_sptrs[i])) // already created in previous run
      if (!_local_output_image_sptrs[i]->has_same_characteristics(*density_info_sptr))
        {
          // previous run was with different sizes, so reallocate
          _local_output_image_sptrs[i].reset(density_info_sptr->get_empty_copy());
        }

#endif
}

void
BackProjectorByBin::check(const ProjDataInfo& proj_data_info) const
{
  if (!this->_already_set_up)
    error("BackProjectorByBin method called without calling set_up first.");
  if (!(*this->_proj_data_info_sptr >= proj_data_info))
    error(format("BackProjectorByBin set-up with different geometry for projection data.\nSet_up was with\n{}\nCalled with\n{}",
                 this->_proj_data_info_sptr->parameter_info(),
                 proj_data_info.parameter_info()));
}

void
BackProjectorByBin::check(const ProjDataInfo& proj_data_info, const DiscretisedDensity<3, float>& density_info) const
{
  this->check(proj_data_info);
  if (!this->_density_sptr->has_same_characteristics(density_info))
    error("BackProjectorByBin set-up with different geometry for density or volume data.");
}

void
BackProjectorByBin::back_project(DiscretisedDensity<3, float>& image, const ProjData& proj_data, int subset_num, int num_subsets)
{
  start_accumulating_in_new_target();
  back_project(proj_data, subset_num, num_subsets);
  get_output(image);
}
#ifdef STIR_PROJECTORS_AS_V3
void
BackProjectorByBin::back_project(DiscretisedDensity<3, float>& image, const RelatedViewgrams<float>& viewgrams)
{
  back_project(image,
               viewgrams,
               viewgrams.get_min_axial_pos_num(),
               viewgrams.get_max_axial_pos_num(),
               viewgrams.get_min_tangential_pos_num(),
               viewgrams.get_max_tangential_pos_num());
}

void
BackProjectorByBin::back_project(DiscretisedDensity<3, float>& image,
                                 const RelatedViewgrams<float>& viewgrams,
                                 const int min_axial_pos_num,
                                 const int max_axial_pos_num)
{
  back_project(image,
               viewgrams,
               min_axial_pos_num,
               max_axial_pos_num,
               viewgrams.get_min_tangential_pos_num(),
               viewgrams.get_max_tangential_pos_num());
}

void
BackProjectorByBin::back_project(DiscretisedDensity<3, float>& density,
                                 const RelatedViewgrams<float>& viewgrams,
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
      error("BackProjectorByBin::back_project called with incorrect related_viewgrams. Problem with symmetries!\n");

    for (RelatedViewgrams<float>::const_iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
      {
        ViewSegmentNumbers vs(iter->get_view_num(), iter->get_segment_num());
        get_symmetries_used()->find_basic_view_segment_numbers(vs);
        // TODOTOF find_basic_view_segment_numbers doesn't fill in timing_pos_num
        vs.timing_pos_num() = basic_vs.timing_pos_num();
        if (vs != basic_vs)
          error("BackProjectorByBin::back_project called with incorrect related_viewgrams. Problem with symmetries!\n");
      }
  }

  actual_back_project(density, viewgrams, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
}
#endif
// -------------------------------------------------------------------------------------------------------------------- //
// The following are repetition of above, where the DiscretisedDensity has already been set with
// start_accumulating_in_new_target()
// -------------------------------------------------------------------------------------------------------------------- //
void
BackProjectorByBin::back_project(const ProjData& proj_data, int subset_num, int num_subsets)
{
  if (!_density_sptr)
    error("You need to call start_accumulating_in_new_target() before back_project()");

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
#ifdef STIR_OPENMP
          RelatedViewgrams<float> viewgrams;
#  pragma omp critical(BACKPROJECTORBYBIN_GETVIEWGRAMS)
          viewgrams = proj_data.get_related_viewgrams(vs, symmetries_sptr, false, k);
          info(format("Processing view {} of segment {}, TOF bin {}", vs.view_num(), vs.segment_num(), k), 3);
#else
          const RelatedViewgrams<float> viewgrams = proj_data.get_related_viewgrams(vs, symmetries_sptr, false, k);
          info(format("Processing view {} of segment {}, TOF bin {}", vs.view_num(), vs.segment_num(), k), 3);
#endif

          back_project(viewgrams);
        }
    }
}

void
BackProjectorByBin::back_project(const RelatedViewgrams<float>& viewgrams)
{
  back_project(viewgrams,
               viewgrams.get_min_axial_pos_num(),
               viewgrams.get_max_axial_pos_num(),
               viewgrams.get_min_tangential_pos_num(),
               viewgrams.get_max_tangential_pos_num());
}

void
BackProjectorByBin::back_project(const RelatedViewgrams<float>& viewgrams,
                                 const int min_axial_pos_num,
                                 const int max_axial_pos_num)
{
  back_project(viewgrams,
               min_axial_pos_num,
               max_axial_pos_num,
               viewgrams.get_min_tangential_pos_num(),
               viewgrams.get_max_tangential_pos_num());
}

void
BackProjectorByBin::back_project(const RelatedViewgrams<float>& viewgrams,
                                 const int min_axial_pos_num,
                                 const int max_axial_pos_num,
                                 const int min_tangential_pos_num,
                                 const int max_tangential_pos_num)
{
  if (viewgrams.get_num_viewgrams() == 0)
    return;

  if (!_density_sptr)
    error("You need to call start_accumulating_in_new_target() before back_project()");

  check(*viewgrams.get_proj_data_info_sptr());

#ifdef STIR_OPENMP
  const int thread_num = omp_get_thread_num();
  if (is_null_ptr(_local_output_image_sptrs[thread_num]))
    _local_output_image_sptrs[thread_num].reset(_density_sptr->get_empty_copy());
#endif

  // first check symmetries
  {
    const ViewSegmentNumbers basic_vs = viewgrams.get_basic_view_segment_num();

    if (get_symmetries_used()->num_related_view_segment_numbers(basic_vs) != viewgrams.get_num_viewgrams())
      error("BackProjectorByBin::back_project called with incorrect related_viewgrams. Problem with symmetries!\n");

    for (RelatedViewgrams<float>::const_iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
      {
        ViewSegmentNumbers vs(iter->get_view_num(), iter->get_segment_num());
        get_symmetries_used()->find_basic_view_segment_numbers(vs);
        // TODOTOF find_basic_view_segment_numbers doesn't fill in timing_pos_num
        vs.timing_pos_num() = basic_vs.timing_pos_num();
        if (vs != basic_vs)
          error("BackProjectorByBin::back_project called with incorrect related_viewgrams. Problem with symmetries!\n");
      }
  }

  actual_back_project(viewgrams, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
}

void
BackProjectorByBin::start_accumulating_in_new_target()
{
  if (!this->_already_set_up)
    error("BackProjectorByBin method called without calling set_up first.");
#ifdef STIR_OPENMP
  if (omp_get_num_threads() != 1)
    error("BackProjectorByBin::start_accumulating_in_new_target cannot be called inside a thread");

  for (int i = 0; i < static_cast<int>(_local_output_image_sptrs.size()); ++i)
    if (!is_null_ptr(_local_output_image_sptrs[i])) // only reset to zero if a thread filled something in
      {
        if (!_local_output_image_sptrs.at(i)->has_same_characteristics(*_density_sptr))
          error("BackProjectorByBin implementation error: local images for openmp have wrong size");
        _local_output_image_sptrs.at(i)->fill(0.F);
      }

#endif
  _density_sptr->fill(0.);
}

void
BackProjectorByBin::get_output(DiscretisedDensity<3, float>& density) const
{
  if (!density.has_same_characteristics(*_density_sptr))
    error("Images should have similar characteristics.");

#ifdef STIR_OPENMP
  if (omp_get_num_threads() != 1)
    error("BackProjectorByBin::get_output() cannot be called inside a thread");

  // "reduce" data constructed by threads
  {
    density.fill(0.F);
    for (int i = 0; i < static_cast<int>(_local_output_image_sptrs.size()); ++i)
      {
        if (!is_null_ptr(_local_output_image_sptrs[i])) // only accumulate if a thread filled something in
          density += *(_local_output_image_sptrs[i]);
      }
  }
#else
  std::copy(_density_sptr->begin_all(), _density_sptr->end_all(), density.begin_all());
#endif

  // If a post-back-projection data processor has been set, apply it.
  if (!is_null_ptr(_post_data_processor_sptr))
    {
      Succeeded success = _post_data_processor_sptr->apply(density);
      if (success != Succeeded::yes)
        throw std::runtime_error("BackProjectorByBin::get_output(). Post-back-projection data processor failed.");
    }
}

void
BackProjectorByBin::set_post_data_processor(shared_ptr<DataProcessor<DiscretisedDensity<3, float>>> post_data_processor_sptr)
{
  _post_data_processor_sptr = post_data_processor_sptr;
}

void
BackProjectorByBin::actual_back_project(
    DiscretisedDensity<3, float>&, const RelatedViewgrams<float>&, const int, const int, const int, const int)
{
  error("BackProjectorByBin::actual_forward_project() This is deprecated and should not be used.");
}

void
BackProjectorByBin::actual_back_project(const RelatedViewgrams<float>& viewgrams,
                                        const int min_axial_pos_num,
                                        const int max_axial_pos_num,
                                        const int min_tangential_pos_num,
                                        const int max_tangential_pos_num)
{
  shared_ptr<DiscretisedDensity<3, float>> density_sptr = _density_sptr;
#ifdef STIR_OPENMP
  const int thread_num = omp_get_thread_num();
  density_sptr = _local_output_image_sptrs[thread_num];
#endif
  actual_back_project(
      *density_sptr, viewgrams, min_axial_pos_num, max_axial_pos_num, min_tangential_pos_num, max_tangential_pos_num);
}

END_NAMESPACE_STIR
