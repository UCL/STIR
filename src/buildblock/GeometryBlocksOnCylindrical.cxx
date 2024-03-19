
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
#include "stir/warning.h"
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
  int num_axial_buckets = scanner.get_num_axial_buckets();
  int num_axial_blocks_per_bucket = scanner.get_num_axial_blocks_per_bucket();
  int num_transaxial_crystals_per_block = scanner.get_num_transaxial_crystals_per_block();
  int num_transaxial_blocks_per_bucket = scanner.get_num_transaxial_blocks_per_bucket();
  int num_transaxial_buckets = scanner.get_num_transaxial_buckets();

  det_pos_to_coord_type cartesian_coord_map_given_detection_position_keys;

  float csi_minus_csiGaps = get_csi_minus_csi_gaps(scanner);

  // calculate first_crystal_offset to build the map
  stir::CartesianCoordinate3D<float> first_crystal_offset(get_initial_axial_z_offset(scanner),
                                                          -scanner.get_effective_ring_radius(),
                                                          get_initial_axial_x_offset_for_each_bucket(scanner));
  int radial_coord = 0;

  // Lopp over axial geometry
  for (int ax_bucket_num = 0; ax_bucket_num < num_axial_buckets; ++ax_bucket_num)
    for (int ax_block_num = 0; ax_block_num < num_axial_blocks_per_bucket; ++ax_block_num)
      for (int ax_crys_num = 0; ax_crys_num < num_axial_crystals_per_block; ++ax_crys_num)
        {
          int axial_coord = get_axial_coord(scanner, ax_bucket_num, ax_block_num, ax_crys_num);
          float axial_translation = get_axial_translation(scanner, ax_bucket_num, ax_block_num, ax_crys_num);

          // Loop over transaxial geometry
          for (int trans_bucket_num = 0; trans_bucket_num < num_transaxial_buckets; ++trans_bucket_num)
            for (int trans_block_num = 0; trans_block_num < num_transaxial_blocks_per_bucket; ++trans_block_num)
              for (int trans_crys_num = 0; trans_crys_num < num_transaxial_crystals_per_block; ++trans_crys_num)
                {
                  // calculate detection position for a given detector
                  // note: in STIR convention, crystal(0,0,0) corresponds to card_coord(z=0,y=-r,x=0)
                  int transaxial_coord = get_transaxial_coord(scanner, trans_bucket_num, trans_block_num, trans_crys_num);
                  stir::DetectionPosition<> det_pos(transaxial_coord, axial_coord, radial_coord);

                  // The translation matrix from the first crystal in the block
                  stir::CartesianCoordinate3D<float> translation_matrix(
                      axial_translation,
                      0.,
                      get_crystal_in_bucket_transaxial_translation(scanner, trans_block_num, trans_crys_num));

                  stir::CartesianCoordinate3D<float> transformed_coord = first_crystal_offset + translation_matrix;

                  // Calculate the rotation of the crystal
                  float alpha = scanner.get_intrinsic_azimuthal_tilt() + trans_bucket_num * (2 * _PI) / num_transaxial_buckets
                                + csi_minus_csiGaps;
                  cartesian_coord_map_given_detection_position_keys[det_pos]
                      = calculate_crystal_rotation(transformed_coord, alpha);
                }
        }
  set_detector_map(cartesian_coord_map_given_detection_position_keys);
}

CartesianCoordinate3D<float>
GeometryBlocksOnCylindrical::calculate_crystal_rotation(const CartesianCoordinate3D<float>& crystal_position,
                                                        const float alpha) const
{
  stir::Array<2, float> rotation_matrix = get_rotation_matrix(alpha);
  // to match index range of CartesianCoordinate3D, which is 1 to 3
  rotation_matrix.set_min_index(1);
  rotation_matrix[1].set_min_index(1);
  rotation_matrix[2].set_min_index(1);
  rotation_matrix[3].set_min_index(1);
  return stir::matrix_multiply(rotation_matrix, crystal_position);
}

int
GeometryBlocksOnCylindrical::get_transaxial_coord(const Scanner& scanner,
                                                  int transaxial_bucket_num,
                                                  int transaxial_block_num,
                                                  int transaxial_crystal_num)
{
  return transaxial_bucket_num * scanner.get_num_transaxial_blocks_per_bucket() * scanner.get_num_transaxial_crystals_per_block()
         + transaxial_block_num * scanner.get_num_transaxial_crystals_per_block() + transaxial_crystal_num;
}

int
GeometryBlocksOnCylindrical::get_axial_coord(const Scanner& scanner,
                                             int axial_bucket_num,
                                             int axial_block_num,
                                             int axial_crystal_num)
{
  return axial_bucket_num * scanner.get_num_axial_blocks_per_bucket() * scanner.get_num_axial_crystals_per_block()
         + axial_block_num * scanner.get_num_axial_crystals_per_block() + axial_crystal_num;
}

float
GeometryBlocksOnCylindrical::get_crystal_in_bucket_transaxial_translation(const Scanner& scanner,
                                                                          int transaxial_block_num,
                                                                          int transaxial_crystal_num)
{
  // Currently, only supports 1 transaxial bucket per angle
  return transaxial_block_num * scanner.get_transaxial_block_spacing()
         + transaxial_crystal_num * scanner.get_transaxial_crystal_spacing();
}

float
GeometryBlocksOnCylindrical::get_axial_translation(const Scanner& scanner,
                                                   int axial_bucket_num,
                                                   int axial_block_num,
                                                   int axial_crystal_num)
{
  return // axial_bucket_num * scanner.get_axial_bucket_spacing() +
      axial_block_num * scanner.get_axial_block_spacing() + axial_crystal_num * scanner.get_axial_crystal_spacing();
}

float
GeometryBlocksOnCylindrical::get_initial_axial_z_offset(const Scanner& scanner)
{
  // Crystals in a block are centered, blocks in a bucket are centered, and buckets are centered in the z axis.
  // This centers the scanner in z
  float crystals_in_block_offset = (scanner.get_num_axial_crystals_per_block() - 1) * scanner.get_axial_crystal_spacing();
  float blocks_in_bucket_offset = (scanner.get_num_axial_blocks_per_bucket() - 1) * scanner.get_axial_block_spacing();
  //  float bucket_offset = (scanner.get_num_axial_buckets() - 1) * scanner.get_axial_bucket_spacing();
  float bucket_offset = 0;

  // Negative because the scanner is centered at z=0 and increases axial coordinates increase
  // 1/2 because it is half the distance from the center to the edge of the scanner
  return -(1.0 / 2) * (crystals_in_block_offset + blocks_in_bucket_offset + bucket_offset);
}

float
GeometryBlocksOnCylindrical::get_initial_axial_x_offset_for_each_bucket(const Scanner& scanner)
{
  // This is the old method... This is probably wrong
  //  float csi_minus_csiGaps = get_csi_minus_csi_gaps(scanner);
  //  float r = scanner.get_effective_ring_radius() / cos(csi_minus_csiGaps);
  //  return -1 * r * sin(csi_minus_csiGaps);

  auto first_crystal_coord = get_crystal_in_bucket_transaxial_translation(scanner, 0, 0);
  auto last_crystal_coord = get_crystal_in_bucket_transaxial_translation(
      scanner, scanner.get_num_transaxial_blocks_per_bucket() - 1, scanner.get_num_transaxial_crystals_per_block() - 1);
  return -(1.0 / 2) * (first_crystal_coord + last_crystal_coord);
}
END_NAMESPACE_STIR
