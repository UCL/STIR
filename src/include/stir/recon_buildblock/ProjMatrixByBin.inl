//
//
/*!

  \file
  \ingroup projection

  \brief Implementations of inline functions for class stir::ProjMatrixByBin

  \author Nikos Efthimiou
  \author Mustapha Sadki 
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2013, Hammersmith Imanet Ltd
    Copyright (C) 2016, University of Hull
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/SymmetryOperation.h"
#include "stir/geometry/line_distances.h"

START_NAMESPACE_STIR

const DataSymmetriesForBins*
ProjMatrixByBin:: get_symmetries_ptr() const
{
  return  symmetries_sptr.get();
}

const shared_ptr<DataSymmetriesForBins>
ProjMatrixByBin:: get_symmetries_sptr() const
{
  return  symmetries_sptr;
}

inline void 
ProjMatrixByBin::
get_proj_matrix_elems_for_one_bin(
                                  ProjMatrixElemsForOneBin& probabilities,
                                  const Bin& bin) STIR_MUTABLE_CONST
{  
  // start_timers(); TODO, can't do this in a const member
  
  // set to empty
  probabilities.erase();
  
  if (cache_stores_only_basic_bins)
  {
    // find basic bin
    Bin basic_bin = bin;    
    unique_ptr<SymmetryOperation> symm_ptr = 
      symmetries_sptr->find_symmetry_operation_from_basic_bin(basic_bin);
    
    probabilities.set_bin(basic_bin);
    // check if basic bin is in cache  
    if (get_cached_proj_matrix_elems_for_one_bin(probabilities) ==
      Succeeded::no)
    {
      // call 'calculate' just for the basic bin
      calculate_proj_matrix_elems_for_one_bin(probabilities);
#ifndef NDEBUG
      probabilities.check_state();
#endif
      cache_proj_matrix_elems_for_one_bin(probabilities);		
    }
    if ( proj_data_info_sptr->is_tof_data() &&
                                 this->tof_enabled)
    {
        LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
        proj_data_info_sptr->get_LOR(lor, bin);
        LORAs2Points<float> lor2(lor);

        Bin fbin = probabilities.get_bin();
        symm_ptr->transform_bin_coordinates(fbin);
        probabilities.set_bin(fbin);

        // now apply TOF kernel and transform to original bin
        apply_tof_kernel_and_symm_transformation(probabilities, lor2.p1(), lor2.p2(), symm_ptr);
    }
    else
    {
        // now transform to original bin
        symm_ptr->transform_proj_matrix_elems_for_one_bin(probabilities);
    }
  }
  else // !cache_stores_only_basic_bins
  {
    probabilities.set_bin(bin);
    // check if in cache  
    if (get_cached_proj_matrix_elems_for_one_bin(probabilities) ==
      Succeeded::no)
    {
      // find basic bin
      Bin basic_bin = bin;  
      unique_ptr<SymmetryOperation> symm_ptr = 
        symmetries_sptr->find_symmetry_operation_from_basic_bin(basic_bin);

      probabilities.set_bin(basic_bin);
      // check if basic bin is in cache
      if (get_cached_proj_matrix_elems_for_one_bin(probabilities) ==
        Succeeded::no)
      {
        // call 'calculate' just for the basic bin
        calculate_proj_matrix_elems_for_one_bin(probabilities);
#ifndef NDEBUG
        probabilities.check_state();
#endif
        cache_proj_matrix_elems_for_one_bin(probabilities);
      }
      if ( proj_data_info_sptr->is_tof_data() &&
                                   this->tof_enabled)
      {
          LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
          proj_data_info_sptr->get_LOR(lor, bin);
          LORAs2Points<float> lor2(lor);

          // now apply TOF kernel and transform to original bin

          Bin fbin = probabilities.get_bin();
          symm_ptr->transform_bin_coordinates(fbin);
          probabilities.set_bin(fbin);
          apply_tof_kernel_and_symm_transformation(probabilities, lor2.p1(), lor2.p2(), symm_ptr);
      }
      else
      {
          // now transform to original bin
          symm_ptr->transform_proj_matrix_elems_for_one_bin(probabilities);
      }
      cache_proj_matrix_elems_for_one_bin(probabilities);      
    }
  }  
  // stop_timers(); TODO, can't do this in a const member
}

void
ProjMatrixByBin::apply_tof_kernel_and_symm_transformation(ProjMatrixElemsForOneBin& tof_probabilities,
                                  const CartesianCoordinate3D<float>& point1,
                                  const CartesianCoordinate3D<float>& point2,
                                  const unique_ptr<SymmetryOperation>& symm_ptr)  STIR_MUTABLE_CONST
{

    CartesianCoordinate3D<float> voxel_center;
    float new_value = 0.f;
    //float low_dist = 0.f;
    //float high_dist = 0.f;

    float d1;
    //float step =  100000.f / 8.f;
    //int p1, p2;

    // THe direction can be from 1 -> 2 depending on the bin sign.
    const CartesianCoordinate3D<float> middle = (point1 + point2)*0.5f;
    const CartesianCoordinate3D<float> difference = point2 - middle;
    const float denom = 1.f / inner_product(difference, difference);
    //    float lor_length = 2.f / (std::sqrt((point1.x() - point2.x()) *(point1.x() - point2.x()) +
    //                                 (point1.y() - point2.y()) *(point1.y() - point2.y()) +
    //                                 (point1.z() - point2.z()) *(point1.z() - point2.z())));

    for (ProjMatrixElemsForOneBin::iterator element_ptr = tof_probabilities.begin();
         element_ptr != tof_probabilities.end(); ++element_ptr)
    {
        Coordinate3D<int> c(element_ptr->get_coords());
        symm_ptr->transform_image_coordinates(c);

        voxel_center =
                image_info_sptr->get_physical_coordinates_for_indices (c);

        /*
         * Original method:
         *
        project_point_on_a_line(point1, point2, voxel_center);

        const CartesianCoordinate3D<float> x = voxel_center - middle;

        const float d1 = inner_product(x, difference) * lor_length;
        */

        // The following is the optimisation of the previous:
        {
            const CartesianCoordinate3D<float> r10 = voxel_center - middle;

            const float u = inner_product(r10, difference) * denom;

            voxel_center[3] = u * difference[3];
            voxel_center[2] = u * difference[2];
            voxel_center[1] = u * difference[1];

            if(u < 0.f)
                d1 =  std::sqrt( voxel_center[3] * voxel_center[3] +
                            voxel_center[2] * voxel_center[2] +
                            voxel_center[1] * voxel_center[1]);
            else
                d1 = -std::sqrt( voxel_center[3] * voxel_center[3] +
                        voxel_center[2] * voxel_center[2] +
                        voxel_center[1] * voxel_center[1]);
        }

        float low_dist = ((proj_data_info_sptr->tof_bin_boundaries_mm[tof_probabilities.get_bin_ptr()->timing_pos_num()].low_lim - d1) * r_sqrt2_gauss_sigma);
        float high_dist = ((proj_data_info_sptr->tof_bin_boundaries_mm[tof_probabilities.get_bin_ptr()->timing_pos_num()].high_lim - d1) * r_sqrt2_gauss_sigma);

        //p1 = (((proj_data_info_sptr->tof_bin_boundaries_mm[tof_probabilities.get_bin_ptr()->timing_pos_num()].low_lim - d1) * r_sqrt2_gauss_sigma) + 4.f) * step;
        //p2 = (((proj_data_info_sptr->tof_bin_boundaries_mm[tof_probabilities.get_bin_ptr()->timing_pos_num()].high_lim - d1) * r_sqrt2_gauss_sigma) + 4.f) * step;

//        if (p1 < 0 || p2 < 0 ||
//                p1 >= 100000 || p2 >= 100000)
//        {
//            *element_ptr = ProjMatrixElemsForOneBin::value_type(c, 0.0f);
//            continue;
//        }

        if (low_dist >= 4.f || high_dist >= 4.f ||
                low_dist <= -4.f || high_dist <= -4.f)
        {
            *element_ptr = ProjMatrixElemsForOneBin::value_type(c, 0.0f);
            continue;
        }

//        get_tof_value(low_dist, high_dist, new_value);
        new_value = element_ptr->get_value() * 0.5f * (erf(high_dist) - erf(low_dist));//*(cache_erf[p2] - cache_erf[p1]); //
        *element_ptr = ProjMatrixElemsForOneBin::value_type(c, new_value);
    }
}

void
ProjMatrixByBin::
//get_tof_value(const float& d1, const float& d2, float& val) const
get_tof_value(const float d1, const float d2, float& val) const
{
    val = 0.5f * (erf(d2) - erf(d1));
}

END_NAMESPACE_STIR
