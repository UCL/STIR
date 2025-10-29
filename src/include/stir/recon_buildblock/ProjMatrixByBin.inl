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

  // The direction can be from 1 -> 2 depending on the bin sign.
  const CartesianCoordinate3D<float> middle = (point1 + point2) * 0.5f;
  const CartesianCoordinate3D<float> diff = point2 - middle;
  const CartesianCoordinate3D<float> diff_unit_vector(diff / static_cast<float>(norm(diff)));

  for (ProjMatrixElemsForOneBin::iterator element_ptr = probabilities.begin(); element_ptr != probabilities.end(); ++element_ptr)
    {
      Coordinate3D<int> c(element_ptr->get_coords());
      const float d2 = -inner_product(image_info_sptr->get_physical_coordinates_for_indices(c) - middle, diff_unit_vector);

      const float low_dist
          = ((proj_data_info_sptr->tof_bin_boundaries_mm[probabilities.get_bin().timing_pos_num()].low_lim - d2));
      const float high_dist
          = ((proj_data_info_sptr->tof_bin_boundaries_mm[probabilities.get_bin().timing_pos_num()].high_lim - d2));

      *element_ptr = ProjMatrixElemsForOneBin::value_type(c, element_ptr->get_value() * get_tof_value(low_dist, high_dist));
    }
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
