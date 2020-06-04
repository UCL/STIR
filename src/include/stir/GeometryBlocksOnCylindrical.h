
/*

TODO copyright and License

*/

/*!
  \file
  \ingroup projdata

  \brief Declaration of class stir::GeometryBlocksOnCylindrical

  \author Parisa Khateri

*/
#ifndef __stir_GeometryBlocksOnCylindrical_H__
#define __stir_GeometryBlocksOnCylindrical_H__

#include "stir/DetectionPosition.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/make_array.h"
#include "stir/numerics/MatrixFunction.h"
#include <map>
#include <string>
#include <vector>
#include <cmath>
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

class GeometryBlocksOnCylindrical
{


public:

	//! Consstructors
	GeometryBlocksOnCylindrical();

	GeometryBlocksOnCylindrical(const shared_ptr<Scanner> &scanner_ptr_v);

	//! Destructor
	~GeometryBlocksOnCylindrical() {}

	//! comparison operator for DetectionPosition class. needded to be used as a key type in building map
	class compare_det_pos{
	public:
		bool operator() (const stir::DetectionPosition<>& , const stir::DetectionPosition<>&) const;
	};

  //! comparison operator for CartesianCoordinate3D class. needded to be used as a key type in building map
 	class compare_cartesian_coord{
	public:
		bool operator() (const stir::CartesianCoordinate3D<float>& , const stir::CartesianCoordinate3D<float>&) const;
	};

	//! Get rotation matrix for a given angle around z axis
	inline stir::Array<2, float> get_rotation_matrix(float alpha);

	//! Build crystal map in cartesian coordinate
	void build_crystal_maps();

	//! Get cartesian coordinate for a given detection position
	inline Succeeded
    find_cartesian_coordinate_given_detection_position(CartesianCoordinate3D<float>& ,
																												DetectionPosition<>);

  //! Get cartesian coordinate for a given detection position
	inline Succeeded
    find_detection_position_given_cartesian_coordinate(DetectionPosition<>&,
																									CartesianCoordinate3D<float>);

	//! Get scanner pointer
	inline const Scanner* get_scanner_ptr() const;


private:
	//! member variables
	shared_ptr<Scanner> scanner_ptr;
	std::map<stir::DetectionPosition<>,
           stir::CartesianCoordinate3D<float>,
           stir::GeometryBlocksOnCylindrical::compare_det_pos> cartesian_coord_map_given_detection_position_keys;
	std::map<stir::CartesianCoordinate3D<float>,
           stir::DetectionPosition<>,
           stir::GeometryBlocksOnCylindrical::compare_cartesian_coord> detection_position_map_given_cartesian_coord_keys_3_decimal;
	std::map<stir::CartesianCoordinate3D<float>,
			 	  stir::DetectionPosition<>,
			 	  stir::GeometryBlocksOnCylindrical::compare_cartesian_coord> detection_position_map_given_cartesian_coord_keys_2_decimal;

};

END_NAMESPACE_STIR

#include "stir/GeometryBlocksOnCylindrical.inl"

#endif
