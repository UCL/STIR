
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
#include "stir/CartesianCoordinate3D.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include <map>
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

/*!
  \ingroup projdata
	\brief A helper class to build the crystal map based on scanner info.

	\This class builds two maps between cartesian coordinates (z, y, x)
	\ and the corresponding detection position (tangential_num, axial_num, radial_num) for each crystal.
	\The crystal map, then, is used in ProjDataInfoBlocksOnCylindrical, ProjDataInfoBlocksOnCylindricalNoArcCorr, and CListRecordSAFIR

	\The center of first ring is the center of coordinates.
	\Distances are from center to center of crystals.

*/

class GeometryBlocksOnCylindrical: public DetectorCoordinateMap
{


public:

	//! Consstructors
	GeometryBlocksOnCylindrical();

	GeometryBlocksOnCylindrical(const shared_ptr<Scanner> &scanner_ptr_v);

	//! Destructor
	~GeometryBlocksOnCylindrical() {}
 private:
	//! Get rotation matrix for a given angle around z axis
	stir::Array<2, float> get_rotation_matrix(float alpha) const;

	//! Build crystal map in cartesian coordinate
	void build_crystal_maps();
 public:
	//! Get cartesian coordinate for a given detection position
	inline Succeeded
          find_cartesian_coordinate_given_detection_position(CartesianCoordinate3D<float>& ,
                                                             const DetectionPosition<>&) const;

        //! Get cartesian coordinate for a given detection position
	Succeeded
          find_detection_position_given_cartesian_coordinate(DetectionPosition<>&,
                                                             const CartesianCoordinate3D<float>&) const;
 private:
	//! Get scanner pointer
	inline const Scanner* get_scanner_ptr() const;


private:
	//! member variables
	shared_ptr<Scanner> scanner_ptr;
	std::map<stir::CartesianCoordinate3D<float>,
          stir::DetectionPosition<>> detection_position_map_given_cartesian_coord_keys_3_decimal;
	std::map<stir::CartesianCoordinate3D<float>,
          stir::DetectionPosition<>> detection_position_map_given_cartesian_coord_keys_2_decimal;

};

END_NAMESPACE_STIR

#include "stir/GeometryBlocksOnCylindrical.inl"

#endif
