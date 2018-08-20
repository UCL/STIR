
/*

TODO copyright and License

*/

/*!
  \file
  \ingroup projdata

  \brief Implementation of inline functions of class stir::GeometryBlocksOnCylindrical

  \author Parisa Khateri

*/

START_NAMESPACE_STIR


stir::Array<2, float>
GeometryBlocksOnCylindrical::
get_rotation_matrix(float alpha)
{
	return stir::make_array(
				stir::make_1d_array(1.F,0.F,0.F),
				stir::make_1d_array(0.F, std::cos(alpha), std::sin(alpha)),
				stir::make_1d_array(0.F, -1*std::sin(alpha), std::cos(alpha))
						);
}

Succeeded
GeometryBlocksOnCylindrical::
find_cartesian_coordinate_given_detection_position(CartesianCoordinate3D<float>& cart_coord,
																				 				DetectionPosition<> det_pos)
{
		if (cartesian_coord_map_given_detection_position_keys.count(det_pos))
		{
			cart_coord = cartesian_coord_map_given_detection_position_keys[det_pos];
			return Succeeded::yes;
		}
		else
		{
			warning("detection position with (tangential_coord, axial_coord, radial_coord)=(%d, %d, %d) does not exist in the inner map",
							det_pos.tangential_coord(), det_pos.axial_coord(), det_pos.radial_coord());
			return Succeeded::no;
		}
}

Succeeded
GeometryBlocksOnCylindrical::
find_detection_position_given_cartesian_coordinate(DetectionPosition<>& det_pos,
																					CartesianCoordinate3D<float> cart_coord)
{
  /*! first round the cartesian coordinates, it might happen that the cart_coord
   is not precisely pointing to the center of the crystal and
   then the det_pos cannot be found using the map
  */
	//rounding cart_coord to 3 decimal place and find det_pos
	cart_coord.z() = (round(cart_coord.z()*1000.0))/1000.0;
	cart_coord.y() = (round(cart_coord.y()*1000.0))/1000.0;
	cart_coord.x() = (round(cart_coord.x()*1000.0))/1000.0;
	if (detection_position_map_given_cartesian_coord_keys_3_decimal.count(cart_coord))
	{
		det_pos =	detection_position_map_given_cartesian_coord_keys_3_decimal[cart_coord];
		return Succeeded::yes;
	}
	else
	{
		//rounding cart_coord to 3 decimal place and find det_pos
		cart_coord.z() = (round(cart_coord.z()*100.0))/100.0;
		cart_coord.y() = (round(cart_coord.y()*100.0))/100.0;
		cart_coord.x() = (round(cart_coord.x()*100.0))/100.0;
		if (detection_position_map_given_cartesian_coord_keys_2_decimal.count(cart_coord))
		{
			det_pos =	detection_position_map_given_cartesian_coord_keys_2_decimal[cart_coord];
			return Succeeded::yes;
		}
		else
		{
			warning("cartesian coordinate (x, y, z)=(%f, %f, %f) does not exist in the inner map",
							cart_coord.x(), cart_coord.y(), cart_coord.z());
			return Succeeded::no;
		}
	}
}

const Scanner*
GeometryBlocksOnCylindrical::
get_scanner_ptr() const
{
	return scanner_ptr.get();
}



END_NAMESPACE_STIR
