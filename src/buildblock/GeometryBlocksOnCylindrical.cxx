
/*

TODO copyright and License

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

START_NAMESPACE_STIR


GeometryBlocksOnCylindrical::
GeometryBlocksOnCylindrical()
{}

GeometryBlocksOnCylindrical::
GeometryBlocksOnCylindrical(const shared_ptr<Scanner> &scanner_ptr_v):
	scanner_ptr(scanner_ptr_v)
{
	build_crystal_maps();
}

bool
GeometryBlocksOnCylindrical::
compare_det_pos::
operator() (const stir::DetectionPosition<>& det_pos1,
						const stir::DetectionPosition<>& det_pos2) const
{
	if ( det_pos1.tangential_coord()<det_pos2.tangential_coord() )
		return 1;
	else if ( det_pos1.tangential_coord()==det_pos2.tangential_coord() && det_pos1.axial_coord()<det_pos2.axial_coord() )
		return 1;
	else if ( det_pos1.tangential_coord()==det_pos2.tangential_coord() && det_pos1.axial_coord()==det_pos2.axial_coord() && det_pos1.radial_coord() < det_pos2.radial_coord() )
		return 1;
	else
		return 0;
}

bool
GeometryBlocksOnCylindrical::
compare_cartesian_coord::
operator() (const stir::CartesianCoordinate3D<float>& cart_coord1,
						const stir::CartesianCoordinate3D<float>& cart_coord2) const
{
	if (cart_coord1.z()<cart_coord2.z())
		return 1;
	else if (cart_coord1.z()==cart_coord2.z() && cart_coord1.y()<cart_coord2.y())
		return 1;
	else if (cart_coord1.z()==cart_coord2.z() && cart_coord1.y()==cart_coord2.y() && cart_coord1.x()<cart_coord2.x())
		return 1;
	else
		return 0;
}

void
GeometryBlocksOnCylindrical::
build_crystal_maps()
{
	// local variables to describe scanner
	int num_axial_crystals_per_block = get_scanner_ptr()->get_num_axial_crystals_per_block();
	int num_transaxial_crystals_per_block = get_scanner_ptr()->get_num_transaxial_crystals_per_block();
	int num_axial_blocks = get_scanner_ptr()->get_num_axial_blocks();
	int num_transaxial_blocks_per_bucket = get_scanner_ptr()->get_num_transaxial_blocks_per_bucket();
	int num_transaxial_buckets = get_scanner_ptr()->get_num_transaxial_blocks()/num_transaxial_blocks_per_bucket;
	int num_detectors_per_ring = get_scanner_ptr()->get_num_detectors_per_ring();
	float axial_block_spacing = get_scanner_ptr()->get_axial_block_spacing();
	float transaxial_block_spacing = get_scanner_ptr()->get_transaxial_block_spacing();
	float axial_crystal_spacing = get_scanner_ptr()->get_axial_crystal_spacing();
	float transaxial_crystal_spacing = get_scanner_ptr()->get_transaxial_crystal_spacing();
	std::string scanner_orientation = get_scanner_ptr()->get_scanner_orientation();

	// check for the scanner orientation
	if (scanner_orientation=="Y" || num_transaxial_buckets%4==0 )
	{/*Building starts from a bucket perpendicular to y axis, from its first crystal.
		see start_x*/

		//calculate start_point to build the map.
		float start_z = -1*(
								((num_axial_blocks-1)/2.)*axial_block_spacing
							+ ((num_axial_crystals_per_block-1)/2.)*axial_crystal_spacing
											 );
		float start_y = -1*get_scanner_ptr()->get_effective_ring_radius();
		float start_x = -1*(
								 ((num_transaxial_blocks_per_bucket-1)/2.)*transaxial_block_spacing
							 + ((num_transaxial_crystals_per_block-1)/2.)*transaxial_crystal_spacing
						 					 ); //the first crystal in the bucket
		stir::CartesianCoordinate3D<float> start_point(start_z, start_y, start_x);

		for (int ax_block_num=0; ax_block_num<num_axial_blocks; ++ax_block_num)
			for (int ax_crys_num=0; ax_crys_num<num_axial_crystals_per_block; ++ax_crys_num)
				for (int trans_bucket_num=0; trans_bucket_num<num_transaxial_buckets; ++trans_bucket_num)
					for (int trans_block_num=0; trans_block_num<num_transaxial_blocks_per_bucket; ++trans_block_num)
						for (int trans_crys_num=0; trans_crys_num<num_transaxial_crystals_per_block; ++trans_crys_num)
		{
			// calculate detection position for a given detector
			// note: in STIR convention, crystal(0,0,0) corresponds to card_coord(z=-l/2,y=-r,x=0)
			int tangential_coord;
			if (num_transaxial_blocks_per_bucket%2==0)
				tangential_coord = trans_bucket_num*num_transaxial_blocks_per_bucket*num_transaxial_crystals_per_block
														 + trans_block_num*num_transaxial_crystals_per_block
														 + trans_crys_num
														 - num_transaxial_blocks_per_bucket/2*num_transaxial_crystals_per_block;
			else
				tangential_coord = trans_bucket_num*num_transaxial_blocks_per_bucket*num_transaxial_crystals_per_block
													 	 + trans_block_num*num_transaxial_crystals_per_block
													 	 + trans_crys_num
														 - num_transaxial_blocks_per_bucket/2*num_transaxial_crystals_per_block
														 - num_transaxial_crystals_per_block/2;

			if (tangential_coord<0)
						tangential_coord += num_detectors_per_ring;

			int axial_coord = ax_block_num*num_axial_crystals_per_block + ax_crys_num;
			int radial_coord = 0;
			stir::DetectionPosition<> det_pos(tangential_coord, axial_coord, radial_coord);

			//calculate cartesion coordinate for a given detector
			stir::CartesianCoordinate3D<float> transformation_matrix(
										ax_block_num*axial_block_spacing + ax_crys_num*axial_crystal_spacing,
										0.,
										trans_block_num*transaxial_block_spacing + trans_crys_num*transaxial_crystal_spacing);
			float alpha = trans_bucket_num*(2*_PI)/num_transaxial_buckets;

			stir::Array<2, float> rotation_matrix = get_rotation_matrix(alpha);
	 		// to match index range of CartesianCoordinate3D, which is 1 to 3
			rotation_matrix.set_min_index(1);
	    rotation_matrix[1].set_min_index(1);
			rotation_matrix[2].set_min_index(1);
			rotation_matrix[3].set_min_index(1);

			stir::CartesianCoordinate3D<float> transformed_coord =
									start_point + transformation_matrix;
			stir::CartesianCoordinate3D<float> cart_coord =
									stir::matrix_multiply(rotation_matrix, transformed_coord);

			// rounding cart_coord to 3 and 2 decimal points then filling maps
			cart_coord.z() = (round(cart_coord.z()*1000.0))/1000.0;
			cart_coord.y() = (round(cart_coord.y()*1000.0))/1000.0;
			cart_coord.x() = (round(cart_coord.x()*1000.0))/1000.0;
			cartesian_coord_map_given_detection_position_keys[det_pos] = cart_coord; //used to find s, m, phi, theta
			detection_position_map_given_cartesian_coord_keys_3_decimal[cart_coord] = det_pos; //used to find bin from listmode data
			cart_coord.z() = (round(cart_coord.z()*100.0))/100.0;
			cart_coord.y() = (round(cart_coord.y()*100.0))/100.0;
			cart_coord.x() = (round(cart_coord.x()*100.0))/100.0;
			detection_position_map_given_cartesian_coord_keys_2_decimal[cart_coord] = det_pos;
		}
	}

	else if (scanner_orientation=="X" )
	{/*Building starts from a bucket perpendicular to x axis, from its first crystal.
		 see start_y*/

		//calculate start_point to build the map.
		float start_z = -1*(
								((num_axial_blocks-1)/2.)*axial_block_spacing
							+ ((num_axial_crystals_per_block-1)/2.)*axial_crystal_spacing
											 );
		float start_x = get_scanner_ptr()->get_effective_ring_radius();
		float start_y = -1*(
								 ((num_transaxial_blocks_per_bucket-1)/2.)*transaxial_block_spacing
							 + ((num_transaxial_crystals_per_block-1)/2.)*transaxial_crystal_spacing
						 					 ); //the first crystal in the bucket
		stir::CartesianCoordinate3D<float> start_point(start_z, start_y, start_x);

		for (int ax_block_num=0; ax_block_num<num_axial_blocks; ++ax_block_num)
			for (int ax_crys_num=0; ax_crys_num<num_axial_crystals_per_block; ++ax_crys_num)
				for (int trans_bucket_num=0; trans_bucket_num<num_transaxial_buckets; ++trans_bucket_num)
					for (int trans_block_num=0; trans_block_num<num_transaxial_blocks_per_bucket; ++trans_block_num)
						for (int trans_crys_num=0; trans_crys_num<num_transaxial_crystals_per_block; ++trans_crys_num)
		{
			// calculate detection position for a given detector
			// note: in STIR convention, crystal(0,0,0) corresponds to (z=-l/2,y=-r,x=0)
			int tangential_coord = trans_bucket_num*num_transaxial_blocks_per_bucket*num_transaxial_crystals_per_block
												 	 + trans_block_num*num_transaxial_crystals_per_block
												   + trans_crys_num;

			int axial_coord = ax_block_num*num_axial_crystals_per_block + ax_crys_num;
			int radial_coord = 0;
			stir::DetectionPosition<> det_pos(tangential_coord, axial_coord, radial_coord);
			//calculate cartesion coordinate for a given detector
			stir::CartesianCoordinate3D<float> transformation_matrix(
								ax_block_num*axial_block_spacing + ax_crys_num*axial_crystal_spacing,
								0.,
								trans_block_num*transaxial_block_spacing + trans_crys_num*transaxial_crystal_spacing);

			float alpha = (trans_bucket_num - num_transaxial_buckets/4)
										*(2*_PI)/num_transaxial_buckets;

			stir::Array<2, float> rotation_matrix = get_rotation_matrix(alpha);
	 		// to match index range of CartesianCoordinate3D, which is 1 to 3
			rotation_matrix.set_min_index(1);
	    rotation_matrix[1].set_min_index(1);
			rotation_matrix[2].set_min_index(1);
			rotation_matrix[3].set_min_index(1);

			stir::CartesianCoordinate3D<float> transformed_coord = start_point+transformation_matrix;
			stir::CartesianCoordinate3D<float> cart_coord =
									stir::matrix_multiply(rotation_matrix, transformed_coord);

			// rounding cart_coord to 3 and 2 decimal points then filling maps
			cart_coord.z() = (round(cart_coord.z()*1000.0))/1000.0;
			cart_coord.y() = (round(cart_coord.y()*1000.0))/1000.0;
			cart_coord.x() = (round(cart_coord.x()*1000.0))/1000.0;
			cartesian_coord_map_given_detection_position_keys[det_pos] = cart_coord; //used to find s, m, phi, theta
			detection_position_map_given_cartesian_coord_keys_3_decimal[cart_coord] = det_pos; //used to find bin from listmode data
			cart_coord.z() = (round(cart_coord.z()*100.0))/100.0;
			cart_coord.y() = (round(cart_coord.y()*100.0))/100.0;
			cart_coord.x() = (round(cart_coord.x()*100.0))/100.0;
			detection_position_map_given_cartesian_coord_keys_2_decimal[cart_coord] = det_pos;
		}
	}
}


END_NAMESPACE_STIR
