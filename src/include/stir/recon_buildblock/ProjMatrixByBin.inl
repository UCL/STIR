//
//
/*!

  \file
  \ingroup projection

  \brief Implementations of inline functions for class stir::ProjMatrixByBin

  \author Nikos Efthimiou
  \author Mustapha Sadki
  \author Kris Thielemans
  \author Robert Twyman
  \author Zekai Li
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2016, University of Hull
    Copyright (C) 2022 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/geometry/line_distances.h"
#include "stir/numerics/erf.h"

START_NAMESPACE_STIR

const DataSymmetriesForBins*
ProjMatrixByBin::get_symmetries_ptr() const
{
  return symmetries_sptr.get();
}

const shared_ptr<DataSymmetriesForBins>
ProjMatrixByBin::get_symmetries_sptr() const
{
  return symmetries_sptr;
}

inline void
ProjMatrixByBin::get_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin& probabilities, const Bin& bin) const
{
  // start_timers(); TODO, can't do this in a const member

  // set to empty
  probabilities.erase();

  if (cache_stores_only_basic_bins)
    {
      // find symmetry operator and basic bin
      Bin basic_bin = bin;
      unique_ptr<SymmetryOperation> symm_ptr = symmetries_sptr->find_symmetry_operation_from_basic_bin(basic_bin);

      probabilities.set_bin(basic_bin);
      // check if basic bin is in cache
      if (get_cached_proj_matrix_elems_for_one_bin(probabilities) == Succeeded::no)
        {
          // basic bin is not in cache, compute lor probabilities for the basic bin
          calculate_proj_matrix_elems_for_one_bin(probabilities);
#ifndef NDEBUG
          probabilities.check_state();
#endif
          if (proj_data_info_sptr->is_tof_data() && this->tof_enabled)
            { // Apply TOF kernel to basic bin
              apply_tof_kernel(probabilities);
            }
          cache_proj_matrix_elems_for_one_bin(probabilities);
        }

      // now transform to original bin (inc. TOF)
      symm_ptr->transform_proj_matrix_elems_for_one_bin(probabilities);
    }
  else
    { // !cache_stores_only_basic_bins
      probabilities.set_bin(bin);
      // if bin is in the cache, get the probabilities
      if (get_cached_proj_matrix_elems_for_one_bin(probabilities) == Succeeded::no)
        {
          // bin probabilities not in the cache, check if basic bins are
          // find basic bin
          Bin basic_bin = bin;
          unique_ptr<SymmetryOperation> symm_ptr = symmetries_sptr->find_symmetry_operation_from_basic_bin(basic_bin);
          probabilities.set_bin(basic_bin);

          // check if basic bin is in cache
          if (get_cached_proj_matrix_elems_for_one_bin(probabilities) == Succeeded::no)
            {
              // basic bin is not in cache, compute lor probabilities for the basic bin
              calculate_proj_matrix_elems_for_one_bin(probabilities);
#ifndef NDEBUG
              probabilities.check_state();
#endif
              if (proj_data_info_sptr->is_tof_data() && this->tof_enabled)
                { // Apply TOF kernel to basic bin
                  apply_tof_kernel(probabilities);
                }
            }
          // now transform basic bin probabilities into original bin probabilities
          symm_ptr->transform_proj_matrix_elems_for_one_bin(probabilities);
          // cache the probabilities for bin
          cache_proj_matrix_elems_for_one_bin(probabilities);
        }
    }
  // stop_timers(); TODO, can't do this in a const member
}

void
ProjMatrixByBin::apply_tof_kernel(ProjMatrixElemsForOneBin& probabilities) const
{
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  proj_data_info_sptr->get_LOR(lor, probabilities.get_bin());
  const LORAs2Points<float> lor2(lor);
  const CartesianCoordinate3D<float> point1 = lor2.p1();
  const CartesianCoordinate3D<float> point2 = lor2.p2();

  // Coordinate system correction: TODO remove in future with ORIGIN shift PR
  // LOR coordinates have origin at scanner center (z=0 at center of all rings)
  // Image coordinates have origin at first ring (z=0 at ring 0)
  // Calculate the offset: distance from first ring to scanner center
  const float scanner_z_offset = (proj_data_info_sptr->get_scanner_ptr()->get_num_rings() - 1) / 2.0f
                                 * proj_data_info_sptr->get_scanner_ptr()->get_ring_spacing();
  const CartesianCoordinate3D<float> coord_system_offset(scanner_z_offset, 0.0f, 0.0f);

  const CartesianCoordinate3D<float> middle = (point1 + point2) * 0.5f;
  const CartesianCoordinate3D<float> diff = point2 - middle;
  const CartesianCoordinate3D<float> diff_unit_vector(diff / static_cast<float>(norm(diff)));

  ProjMatrixElemsForOneBin tof_row(probabilities.get_bin());
  // Estimate size of TOF row such that we can pre-allocate.
  std::size_t max_num_elements;
  {
    const auto length = norm(point1 - point2);
    const auto kernel_width = 8 / r_sqrt2_gauss_sigma;
    const auto tof_bin_width = proj_data_info_sptr->tof_bin_boundaries_mm[probabilities.get_bin().timing_pos_num()].high_lim
                               - proj_data_info_sptr->tof_bin_boundaries_mm[probabilities.get_bin().timing_pos_num()].low_lim;
    const auto fraction = (kernel_width + tof_bin_width) / length;
    // This seems to sometimes over-, sometimes underestimate, but it's probably not very important
    // as std::vector will grow as necessary.
    max_num_elements = std::size_t(probabilities.size() * std::min(fraction * 1.2, 1.001));
  }
  tof_row.reserve(max_num_elements);

  for (ProjMatrixElemsForOneBin::iterator element_ptr = probabilities.begin(); element_ptr != probabilities.end(); ++element_ptr)
    {
      Coordinate3D<int> c(element_ptr->get_coords());
      // Get voxel physical coordinates (in image coordinate system)
      const CartesianCoordinate3D<float> voxel_pos_image = image_info_sptr->get_physical_coordinates_for_indices(c);

      // Convert to scanner coordinate system by subtracting the offset
      const CartesianCoordinate3D<float> voxel_pos_scanner = voxel_pos_image - coord_system_offset;

      // Now compute TOF distance in the same coordinate system as the LOR
      const float d2 = -inner_product(voxel_pos_scanner - middle, diff_unit_vector);

      const float low_dist
          = ((proj_data_info_sptr->tof_bin_boundaries_mm[probabilities.get_bin().timing_pos_num()].low_lim - d2));
      const float high_dist
          = ((proj_data_info_sptr->tof_bin_boundaries_mm[probabilities.get_bin().timing_pos_num()].high_lim - d2));

      const auto tof_kernel_value = get_tof_value(low_dist, high_dist);
      if (tof_kernel_value > 0)
        {
          if (auto non_tof_value = element_ptr->get_value())
            tof_row.push_back(ProjMatrixElemsForOneBin::value_type(c, non_tof_value * tof_kernel_value));
        }
      else
        {
          // Optimisation would be to get out of the loop once we're "beyond" the TOF kernel,
          // but it is tricky to do. It requires that the input is sorted
          // "along" the LOR, i.e. d2 increases montonically, but that doesn't seem to be true.
          // if (tof_row.size() > 0)
          //   break;
        }
    }
  probabilities = tof_row;
  // info("Estimate " + std::to_string(max_num_elements) + ", actual " + std::to_string(tof_row.size()));
}

float
ProjMatrixByBin::get_tof_value(const float d1, const float d2) const
{
  const float d1_n = d1 * r_sqrt2_gauss_sigma;
  const float d2_n = d2 * r_sqrt2_gauss_sigma;

  if ((d1_n >= 4.f && d2_n >= 4.f) || (d1_n <= -4.f && d2_n <= -4.f))
    return 0.F;
  else
    return static_cast<float>(0.5 * (erf_interpolation(d2_n) - erf_interpolation(d1_n)));
}

END_NAMESPACE_STIR
