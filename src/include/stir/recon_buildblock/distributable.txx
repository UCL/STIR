/*
    Copyright (C) 2024, 2025 University College London
    Copyright (C) 2020, 2022, University of Pennsylvania
    Copyright (C) 2025, Commonwealth Scientific and Industrial Research Organisation
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup distributable

  \brief Implementation of stir::LM_distributable_computation() and related functions

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author Ashley Gillman
*/
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/distributable.h"
#include "stir/DiscretisedDensity.h"
#include "stir/CPUTimer.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include "stir/format.h"

#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/Bin.h"

#include "stir/num_threads.h"

START_NAMESPACE_STIR

template <typename CallBackT>
void
LM_distributable_computation(const shared_ptr<ProjMatrixByBin> PM_sptr,
                             const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
                             DiscretisedDensity<3, float>* output_image_ptr,
                             const DiscretisedDensity<3, float>* input_image_ptr,
                             const std::vector<BinAndCorr>& record_ptr,
                             const int subset_num,
                             const int num_subsets,
                             const bool has_add,
                             const bool accumulate,
                             double* double_out_ptr,
                             CallBackT&& call_back)
{

  CPUTimer CPU_timer;
  CPU_timer.start();
  HighResWallClockTimer wall_clock_timer;
  wall_clock_timer.start();

  assert(!record_ptr.empty());

  if (output_image_ptr != NULL && !accumulate)
    output_image_ptr->fill(0.F);

  std::vector<shared_ptr<DiscretisedDensity<3, float>>> local_output_image_sptrs;
  std::vector<double> local_double_outs;
  std::vector<double*> local_double_out_ptrs;
  std::vector<int> local_counts, local_count2s;
  std::vector<ProjMatrixElemsForOneBin> local_row;
#ifdef STIR_OPENMP
#  pragma omp parallel shared(local_output_image_sptrs, local_row, local_double_outs, local_counts, local_count2s)
#endif
  // start of threaded section if openmp
  {
#ifdef STIR_OPENMP
#  pragma omp single
#endif
    // allocate "local" vectors
    {
#ifdef STIR_OPENMP
      const auto num_threads = omp_get_num_threads();
#else
      const int num_threads = 1;
#endif
      info("Listmode gradient calculation: starting loop with " + std::to_string(num_threads) + " threads", 2);
      local_output_image_sptrs.resize(get_max_num_threads(), shared_ptr<DiscretisedDensity<3, float>>());
      local_double_out_ptrs.resize(get_max_num_threads(), 0);
      if (double_out_ptr)
        {
          local_double_outs.resize(get_max_num_threads(), 0.);
          for (int t = 0; t < get_max_num_threads(); ++t)
            local_double_out_ptrs[t] = &local_double_outs[t];
        }
      local_counts.resize(get_max_num_threads(), 0);
      local_count2s.resize(get_max_num_threads(), 0);
      local_row.resize(get_max_num_threads(), ProjMatrixElemsForOneBin());
    }
#ifdef STIR_OPENMP
#  pragma omp for schedule(dynamic)
#endif
    // note: VC uses OpenMP 2.0, so need signed integer for loop
    for (long int ievent = 0; ievent < static_cast<long>(record_ptr.size()); ++ievent)
      {
        auto& record = record_ptr.at(ievent);
        if (record.my_bin.get_bin_value() == 0.0f) // shouldn't happen really, but a check probably doesn't hurt
          continue;

#ifdef STIR_OPENMP
        const int thread_num = omp_get_thread_num();
#else
        const int thread_num = 0;
#endif

        if (output_image_ptr != NULL)
          {
            if (is_null_ptr(local_output_image_sptrs[thread_num]))
              local_output_image_sptrs[thread_num].reset(output_image_ptr->get_empty_copy());
          }

        const Bin& measured_bin = record.my_bin;

        if (num_subsets > 1)
          {
            Bin basic_bin = measured_bin;
            if (!PM_sptr->get_symmetries_ptr()->is_basic(measured_bin))
              PM_sptr->get_symmetries_ptr()->find_basic_bin(basic_bin);

            if (subset_num != static_cast<int>(basic_bin.view_num() % num_subsets))
              {
                continue;
              }
          }

        PM_sptr->get_proj_matrix_elems_for_one_bin(local_row[thread_num], measured_bin);
        call_back(*local_output_image_sptrs[thread_num],
                  local_row[thread_num],
                  has_add ? record.my_corr : 0.F,
                  measured_bin,
                  *input_image_ptr,
                  local_double_out_ptrs[thread_num]);
      }
  }
  // flatten data constructed by threads (or collapse unitary dim if no threading)
  {
    if (double_out_ptr != NULL)
      {
        for (int i = 0; i < static_cast<int>(local_double_outs.size()); ++i)
          *double_out_ptr += local_double_outs[i]; // accumulate all (as they were initialised to zero)
      }
    // count += std::accumulate(local_counts.begin(), local_counts.end(), 0);
    // count2 += std::accumulate(local_count2s.begin(), local_count2s.end(), 0);

    if (output_image_ptr != NULL)
      {
        for (int i = 0; i < static_cast<int>(local_output_image_sptrs.size()); ++i)
          if (!is_null_ptr(local_output_image_sptrs[i])) // only accumulate if a thread filled something in
            *output_image_ptr += *(local_output_image_sptrs[i]);
      }
  }
  CPU_timer.stop();
  wall_clock_timer.stop();
  info(format(
      "Computation times for distributable_computation, CPU {}s, wall-clock {}s", CPU_timer.value(), wall_clock_timer.value()));
}

END_NAMESPACE_STIR
