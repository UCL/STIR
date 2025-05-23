/* 
	Copyright 2015, 2017 ETH Zurich, Institute of Particle Physics
    Copyright (C) 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock
  \brief Implementation of class stir::DetectorCoordinateMap

  \author Jannis Fischer
  \author Parisa Khateri
  \author Michael Roethlisberger
  \author Kris Thielemans
*/

#include "stir/error.h"
#include "stir/DetectorCoordinateMap.h"
#include "stir/modulo.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
	
DetectorCoordinateMap::det_pos_to_coord_type DetectorCoordinateMap::read_detectormap_from_file_help( const std::string& filename )
{
	std::ifstream myfile(filename.c_str());
	if( !myfile ) 
	{
		error("Error opening file '" + filename + "'");
	}

    det_pos_to_coord_type coord_map;
	std::string line;
	while( std::getline( myfile, line))
	{
		if( line.size() && line[0] == '#' ) continue;
		bool has_layer_index = false;
		stir::CartesianCoordinate3D<float> coord;
		stir::DetectionPosition<> detpos;
		std::vector<std::string> col;
		boost::split(col, line, boost::is_any_of("\t,"));
		if( !col.size() ) break;
		else if( col.size() == 5 ) has_layer_index = false;
		else if( col.size() == 6 ) has_layer_index = true;
		coord[1] = static_cast<float>(atof(col[4+has_layer_index].c_str() ));
		coord[2] = static_cast<float>(atof(col[3+has_layer_index].c_str() ));
		coord[3] = static_cast<float>(atof(col[2+has_layer_index].c_str() ));
		
		if( !has_layer_index ) detpos.radial_coord() = 0;
		else detpos.radial_coord() = atoi(col[2].c_str());
		detpos.axial_coord() = atoi(col[0].c_str());
		detpos.tangential_coord() = atoi(col[1].c_str());

		coord_map[detpos] = coord;
	}
	return coord_map;
}


void DetectorCoordinateMap::set_detector_map( const DetectorCoordinateMap::det_pos_to_coord_type& coord_map )
{
    // The detector crystal coordinates are saved in coord_map the following way:
    // (detector#, ring#, 1)[(x,y,z)]
    // the detector# and ring# are determined outside of STIR (later given in input)
    // In order to fulfill the STIR convention we have to give the coordinates  
    // detector# and ring# defined by ourself so that the start (0,0) goes to the 
    // coordinate with the smallest z and smallest y and the detector# is  
    // counterclockwise rising.
    // To achieve this, we assign each coordinate the value 'coord_sorter' which
    // is the assigned value of the criteria mentioned above. With it we sort the 
    // coordinates and fill the to maps 'input_index_to_det_pos' and 
    // 'det_pos_to_coord'.
    std::vector<double> coords_to_be_sorted;
    boost::unordered_map<double, stir::DetectionPosition<> > map_for_sorting_coordinates;
    coords_to_be_sorted.reserve(coord_map.size());

    unsigned min_tangential_coord = 1000000U;
    unsigned min_axial_coord = 1000000U;
    unsigned min_radial_coord = 1000000U;
    num_tangential_coords = 0U;
    num_axial_coords = 0U;
    num_radial_coords = 0U;
    for(auto it : coord_map)
    {
        double coord_sorter = it.second[1] * 100 + from_min_pi_plus_pi_to_0_2pi(std::atan2(it.second[3], -it.second[2]));
        coords_to_be_sorted.push_back(coord_sorter);
        map_for_sorting_coordinates[coord_sorter] = it.first;

        const auto detpos = it.first;
        if (num_tangential_coords <= detpos.tangential_coord())
          num_tangential_coords = detpos.tangential_coord()+1;
        if (min_tangential_coord > detpos.tangential_coord())
          min_tangential_coord = detpos.tangential_coord();
        if (num_axial_coords <= detpos.axial_coord())
          num_axial_coords = detpos.axial_coord()+1;
        if (min_axial_coord > detpos.axial_coord())
          min_axial_coord = detpos.axial_coord();
        if (num_radial_coords <= detpos.radial_coord())
          num_radial_coords = detpos.radial_coord()+1;
        if (min_radial_coord > detpos.radial_coord())
          min_radial_coord = detpos.radial_coord(); 
    }

    if ((min_tangential_coord != 0) || (min_axial_coord != 0) || (min_radial_coord != 0))
      error("DetectorCoordinateMap::set_detector_map: minimum indices have to be zero.");
    if ((num_tangential_coords * num_axial_coords * num_radial_coords) != coord_map.size())
        error("DetectorCoordinateMap::set_detector_map: maximum indices inconsistent with a regular 3D array.\n"
              "Sizes derived from indices: tangential " + std::to_string(num_tangential_coords) +
              ", axial " +std::to_string(num_axial_coords) + ", radial " + std::to_string(num_radial_coords) +
              "\nOveral  size: " + std::to_string(coord_map.size()));

//    std::sort(coords_to_be_sorted.begin(), coords_to_be_sorted.end());
    stir::DetectionPosition<> detpos(0,0,0);
    for(std::vector<double>::iterator it = coords_to_be_sorted.begin(); it != coords_to_be_sorted.end();++it)
      {
#if 0
        input_index_to_det_pos[map_for_sorting_coordinates[*it]] = detpos;
        auto cart_coord = coord_map.at(map_for_sorting_coordinates[*it]);
#else
        input_index_to_det_pos[detpos] = detpos;
        auto cart_coord = coord_map.at(detpos);
#endif
        // rounding cart_coord to 3 and 2 decimal points then filling maps
        cart_coord.z() = (round(cart_coord.z()*1000.0F))/1000.0F;
        cart_coord.y() = (round(cart_coord.y()*1000.0F))/1000.0F;
        cart_coord.x() = (round(cart_coord.x()*1000.0F))/1000.0F;
	det_pos_to_coord[detpos] = cart_coord;
        detection_position_map_given_cartesian_coord_keys_3_decimal[cart_coord] = detpos; //used to find bin from listmode data
        cart_coord.z() = (round(cart_coord.z()*100.0F))/100.0F;
        cart_coord.y() = (round(cart_coord.y()*100.0F))/100.0F;
        cart_coord.x() = (round(cart_coord.x()*100.0F))/100.0F;
        detection_position_map_given_cartesian_coord_keys_2_decimal[cart_coord] = detpos;

        ++detpos.tangential_coord();
        if (detpos.tangential_coord() == num_tangential_coords)
          {
            ++detpos.axial_coord();
            detpos.tangential_coord() = 0;
            if (detpos.axial_coord() == num_axial_coords)
              {
                ++detpos.radial_coord();
                detpos.axial_coord() = 0;
              }
          }
      }
}

// creates maps to convert between stir and 3d coordinates
void DetectorCoordinateMap::read_detectormap_from_file( const std::string& filename )
{
  det_pos_to_coord_type coord_map =
  read_detectormap_from_file_help(filename);
  set_detector_map(coord_map);
}

Succeeded
DetectorCoordinateMap::
find_detection_position_given_cartesian_coordinate(DetectionPosition<>& det_pos,
                                                   const CartesianCoordinate3D<float>& cart_coord) const
{
  /*! first round the cartesian coordinates, it might happen that the cart_coord
   is not precisely pointing to the center of the crystal and
   then the det_pos cannot be found using the map
  */
  //rounding cart_coord to 3 decimal place and find det_pos
  CartesianCoordinate3D<float> rounded_cart_coord;
  rounded_cart_coord.z() = round(cart_coord.z()*1000.0F)/1000.0F;
  rounded_cart_coord.y() = round(cart_coord.y()*1000.0F)/1000.0F;
  rounded_cart_coord.x() = round(cart_coord.x()*1000.0F)/1000.0F;
	if (detection_position_map_given_cartesian_coord_keys_3_decimal.count(rounded_cart_coord))
	{
          det_pos =	detection_position_map_given_cartesian_coord_keys_3_decimal.at(rounded_cart_coord);
          return Succeeded::yes;
	}
	else
	{
          //rounding cart_coord to 2 decimal place and find det_pos
		rounded_cart_coord.z() = round(cart_coord.z()*100.0F)/100.0F;
		rounded_cart_coord.y() = round(cart_coord.y()*100.0f)/100.0F;
		rounded_cart_coord.x() = round(cart_coord.x()*100.0F)/100.0F;
		if (detection_position_map_given_cartesian_coord_keys_2_decimal.count(rounded_cart_coord))
		{
                  det_pos =	detection_position_map_given_cartesian_coord_keys_2_decimal.at(rounded_cart_coord);
                  return Succeeded::yes;
		}
		else
		{
			rounded_cart_coord.z() = round(cart_coord.z()*10.0F)/10.0F;
			rounded_cart_coord.y() = round(cart_coord.y()*10.0f)/10.0F;
			rounded_cart_coord.x() = round(cart_coord.x()*10.0F)/10.0F;
			if (detection_position_map_given_cartesian_coord_keys_2_decimal.count(rounded_cart_coord))
			{
                  det_pos =	detection_position_map_given_cartesian_coord_keys_2_decimal.at(rounded_cart_coord);
                  return Succeeded::yes;
			}else{
				warning("cartesian coordinate (x, y, z)=(%f, %f, %f) does not exist in the inner map",
							cart_coord.x(), cart_coord.y(), cart_coord.z());
				return Succeeded::no;
			}
		}
	}
}

END_NAMESPACE_STIR
