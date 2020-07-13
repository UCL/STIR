
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

  \brief Implementation of inline functions of class stir::GeometryBlocksOnCylindrical

  \author Parisa Khateri

*/

START_NAMESPACE_STIR

Succeeded
GeometryBlocksOnCylindrical::
find_cartesian_coordinate_given_detection_position(CartesianCoordinate3D<float>& cart_coord,
                                                   const DetectionPosition<>& det_pos) const
{
  if (cartesian_coord_map_given_detection_position_keys.count(det_pos))
    {
      cart_coord = cartesian_coord_map_given_detection_position_keys.at(det_pos);
      return Succeeded::yes;
    }
  else
    {
      warning("detection position with (tangential_coord, axial_coord, radial_coord)=(%d, %d, %d) does not exist in the inner map",
              det_pos.tangential_coord(), det_pos.axial_coord(), det_pos.radial_coord());
      return Succeeded::no;
    }
}

const Scanner*
GeometryBlocksOnCylindrical::
get_scanner_ptr() const
{
	return scanner_ptr.get();
}



END_NAMESPACE_STIR
