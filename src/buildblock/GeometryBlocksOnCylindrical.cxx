
/*
Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*!
  \file
  \ingroup projdata

  \brief  Non-inline implementations of stir::GeometryBlocksOnCylindrical

  \author Parisa Khateri

*/

#include "stir/DetectionPosition.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/GeometryBlocksOnCylindrical.h"
#include <string>
#include <cmath>
#include "stir/Array.h"
#include "stir/make_array.h"
#include "stir/numerics/MatrixFunction.h"
#include <map>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <boost/format.hpp>

START_NAMESPACE_STIR

GeometryBlocksOnCylindrical::GeometryBlocksOnCylindrical(const Scanner& scanner)
{
  if (scanner.check_consistency() == Succeeded::no)
    error("Error in GeometryBlocksOnCylindrical: scanner configuration not accepted. Please check warnings.");
  build_crystal_maps(scanner);
}

stir::Array<2, float>
GeometryBlocksOnCylindrical::get_rotation_matrix(float alpha) const
{
  return stir::make_array(stir::make_1d_array(1.F, 0.F, 0.F),
                          stir::make_1d_array(0.F, std::cos(alpha), std::sin(alpha)),
                          stir::make_1d_array(0.F, -1 * std::sin(alpha), std::cos(alpha)));
}

void
GeometryBlocksOnCylindrical::build_crystal_maps(const Scanner& scanner)
{
  // local variables to describe scanner
  int num_axial_crystals_per_block = scanner.get_num_axial_crystals_per_block();
  int num_transaxial_crystals_per_block = scanner.get_num_transaxial_crystals_per_block();
  int num_transaxial_blocks_per_bucket = scanner.get_num_transaxial_blocks_per_bucket();
  int num_axial_blocks_per_bucket = scanner.get_num_axial_blocks_per_bucket();
  int num_transaxial_buckets = scanner.get_num_transaxial_buckets();
  int num_axial_buckets = scanner.get_num_axial_buckets();
  int num_detectors_per_ring = scanner.get_num_detectors_per_ring();
  float axial_block_spacing = scanner.get_axial_block_spacing();
  float transaxial_block_spacing = scanner.get_transaxial_block_spacing();
  float axial_crystal_spacing = scanner.get_axial_crystal_spacing();
  float transaxial_crystal_spacing = scanner.get_transaxial_crystal_spacing();

  det_pos_to_coord_type cartesian_coord_map_given_detection_position_keys;
  /*Building starts from a bucket perpendicular to y axis, from its first crystal.
          see start_x*/

  // calculate start_point to build the map.

  //    estimate the angle covered by half bucket, csi
  float csi = _PI / num_transaxial_buckets;
  float trans_blocks_gap = transaxial_block_spacing - num_transaxial_crystals_per_block * transaxial_crystal_spacing;
  float ax_blocks_gap = axial_block_spacing - (num_axial_crystals_per_block - 1) * axial_crystal_spacing;
  float csi_minus_csiGaps = csi - (csi / transaxial_block_spacing * 2) * (transaxial_crystal_spacing / 2 + trans_blocks_gap);
  float start_z = -(axial_block_spacing * num_axial_blocks_per_bucket * num_axial_buckets - ax_blocks_gap) / 2;
  float start_y = -1 * scanner.get_effective_ring_radius();
  float start_x = -1 // the first crystal in the bucket
                  * (((num_transaxial_blocks_per_bucket - 1) / 2.) * transaxial_block_spacing
                     + ((num_transaxial_crystals_per_block - 1) / 2.) * transaxial_crystal_spacing);

  stir::CartesianCoordinate3D<float> start_point(start_z, start_y, start_x);

  for (int ax_bucket_num = 0; ax_bucket_num < num_axial_buckets; ++ax_bucket_num)
    for (int ax_block_num = 0; ax_block_num < num_axial_blocks_per_bucket; ++ax_block_num)
      for (int ax_crys_num = 0; ax_crys_num < num_axial_crystals_per_block; ++ax_crys_num)
        for (int trans_bucket_num = 0; trans_bucket_num < num_transaxial_buckets; ++trans_bucket_num)
          for (int trans_block_num = 0; trans_block_num < num_transaxial_blocks_per_bucket; ++trans_block_num)
            for (int trans_crys_num = 0; trans_crys_num < num_transaxial_crystals_per_block; ++trans_crys_num)
              {
                // calculate detection position for a given detector
                // note: in STIR convention, crystal(0,0,0) corresponds to card_coord(z=0,y=-r,x=0)
                int tangential_coord;
                tangential_coord = trans_bucket_num * num_transaxial_blocks_per_bucket * num_transaxial_crystals_per_block
                                   + trans_block_num * num_transaxial_crystals_per_block + trans_crys_num;

                if (tangential_coord < 0)
                  tangential_coord += num_detectors_per_ring;

                int axial_coord = ax_bucket_num * num_axial_blocks_per_bucket * num_axial_crystals_per_block
                                  + ax_block_num * num_axial_crystals_per_block + ax_crys_num;
                int radial_coord = 0;
                stir::DetectionPosition<> det_pos(tangential_coord, axial_coord, radial_coord);

                // calculate cartesian coordinate for a given detector
                stir::CartesianCoordinate3D<float> transformation_matrix(
                    (ax_block_num + ax_bucket_num * num_axial_blocks_per_bucket) * axial_block_spacing
                        + ax_crys_num * axial_crystal_spacing,
                    0.,
                    trans_block_num * transaxial_block_spacing + trans_crys_num * transaxial_crystal_spacing);
                float alpha = scanner.get_intrinsic_azimuthal_tilt() + trans_bucket_num * (2 * _PI) / num_transaxial_buckets
                              + csi_minus_csiGaps;

                stir::Array<2, float> rotation_matrix = get_rotation_matrix(alpha);
                // to match index range of CartesianCoordinate3D, which is 1 to 3
                rotation_matrix.set_min_index(1);
                rotation_matrix[1].set_min_index(1);
                rotation_matrix[2].set_min_index(1);
                rotation_matrix[3].set_min_index(1);

                stir::CartesianCoordinate3D<float> transformed_coord = start_point + transformation_matrix;
                stir::CartesianCoordinate3D<float> cart_coord = stir::matrix_multiply(rotation_matrix, transformed_coord);

                cartesian_coord_map_given_detection_position_keys[det_pos] = cart_coord;
              }
  set_detector_map(cartesian_coord_map_given_detection_position_keys);
}

END_NAMESPACE_STIR
