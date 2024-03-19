
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

  \brief Declaration of class stir::GeometryBlocksOnCylindrical

  \author Parisa Khateri

*/
#ifndef __stir_GeometryBlocksOnCylindrical_H__
#define __stir_GeometryBlocksOnCylindrical_H__

#include "stir/DetectorCoordinateMap.h"
#include "stir/Array.h"

START_NAMESPACE_STIR

/*!
  \ingroup projdata
        \brief A helper class to build the crystal map based on scanner info.

        This class builds two maps between cartesian coordinates (z, y, x)
        and the corresponding detection position (tangential_num, axial_num, radial_num) for each crystal.
        The crystal map, then, is used in ProjDataInfoBlocksOnCylindrical, ProjDataInfoBlocksOnCylindricalNoArcCorr, and
  CListRecordSAFIR

        The center of first ring is the center of coordinates.
        Distances are from center to center of crystals.

*/

class GeometryBlocksOnCylindrical : public DetectorCoordinateMap
{

public:
  GeometryBlocksOnCylindrical(const Scanner& scanner);

  //! Calculates the transaxial coordinate of a crystal given a scanner and the crystal's indices
  static int
  get_transaxial_coord(const Scanner& scanner, int transaxial_bucket_num, int transaxial_block_num, int transaxial_crystal_num);

  //! Calculates the axial coordinate of a crystal given a scanner and the crystal's indices
  static int get_axial_coord(const Scanner& scanner, int axial_bucket_num, int axial_block_num, int axial_crystal_num);

  //! Calculates the transaxial translation of a crystal given a scanner and the crystal's indices
  static float
  get_crystal_in_bucket_transaxial_translation(const Scanner& scanner, int transaxial_block_num, int transaxial_crystal_num);

  //! Calculates the axial translation of a crystal given a scanner and the crystal's indices
  static float get_axial_translation(const Scanner& scanner, int axial_bucket_num, int axial_block_num, int axial_crystal_num);

  //! Calculate the initial axial z offset to center the scanner on 0,0,0
  static float get_initial_axial_z_offset(const Scanner& scanner);

  //! Calculate the initial transaxial x offset to center the scanner on 0,0,0
  static float get_initial_axial_x_offset_for_each_bucket(const Scanner& scanner);

  static float get_csi_minus_csi_gaps(const Scanner& scanner)
  {
    //! Calculate the CSI, angle covered by half a bucket
    // 2 * PI / num_transaxial_buckets / 2 (simplified)
    float csi = _PI / scanner.get_num_transaxial_buckets(); // TODO, this assumes 1 transaxial bucket per angle

    // The difference between the transaxial block spacing and the sum of all transaxial crystal spacing's in the block
    float trans_blocks_gap = scanner.get_transaxial_block_spacing()
                             - scanner.get_num_transaxial_crystals_per_block() * scanner.get_transaxial_crystal_spacing();
    // Calculate the angle covered by the gaps between the blocks
    float csi_gaps
        = 2 * csi * (scanner.get_transaxial_crystal_spacing() / 2 + trans_blocks_gap) / scanner.get_transaxial_block_spacing();
    return csi - csi_gaps;
  };

  //!
  CartesianCoordinate3D<float> calculate_crystal_rotation(const CartesianCoordinate3D<float>& crystal_position,
                                                          const float alpha) const;

private:
  //! Get rotation matrix for a given angle around z axis
  stir::Array<2, float> get_rotation_matrix(float alpha) const;

  //! Build crystal map in cartesian coordinate
  void build_crystal_maps(const Scanner& scanner);
};

END_NAMESPACE_STIR

#endif
