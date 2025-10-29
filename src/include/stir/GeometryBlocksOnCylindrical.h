
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

private:
  //! Get rotation matrix for a given angle around z axis
  stir::Array<2, float> get_rotation_matrix(float alpha) const;

  //! Build crystal map in cartesian coordinate
  void build_crystal_maps(const Scanner& scanner);
};

END_NAMESPACE_STIR

#endif
