/*
	Copyright 2015, 2017 ETH Zurich, Institute of Particle Physics
	Copyright 2020 Positrigo AG, Zurich
    Copyright (C) 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
 */

/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::DetectorCoordinateMap

  \author Jannis Fischer
  \author Parisa Khateri
  \author Kris Thielemans
*/

#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <map>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>

#include "stir/CartesianCoordinate3D.h"
#include "stir/DetectionPosition.h"

#ifndef __stir_DetectorCoordinateMap_H__
#define __stir_DetectorCoordinateMap_H__

START_NAMESPACE_STIR

class Succeeded;

/*! Class providing map functionality to convert detector indices to spatial coordinates. 
	Map files can have 5 or 6 tab- or comma-separated columns. Lines beginning with '#' are ignored. The layer column is optional
	\par Format:
	ring,detector,(layer,)x,y,z
	An empty line will terminate the reading at that line.

    Optionally LOR end-points can be randomly displaced using a Gaussian distribution with standard deviation \sigma (in mm).
*/
class DetectorCoordinateMap
{
	struct ihash
	    : std::unary_function<stir::DetectionPosition<> , std::size_t>
	{
	    std::size_t operator()(stir::DetectionPosition<>  const& detpos) const
	    {
		    std::size_t seed = 0;
		    boost::hash_combine(seed, detpos.axial_coord());
		    boost::hash_combine(seed, detpos.radial_coord());
		    boost::hash_combine(seed, detpos.tangential_coord());
		    return seed;
	    }
	};
public:
    typedef boost::unordered_map<stir::DetectionPosition<>, stir::CartesianCoordinate3D<float>, ihash> det_pos_to_coord_type;
    typedef boost::unordered_map<stir::DetectionPosition<>, stir::DetectionPosition<>, ihash> unordered_to_ordered_det_pos_type;

	//! Constructor calls read_detectormap_from_file( filename ).
	DetectorCoordinateMap(const std::string& filename, double sigma = 0.0) :
		sigma(sigma),
		distribution(0.0, sigma)
		{ read_detectormap_from_file( filename ); }
	//! Constructor calls set_detector_map(coord_map).
	DetectorCoordinateMap(const det_pos_to_coord_type& coord_map, double sigma = 0.0) :
		sigma(sigma),
		distribution(0.0, sigma)
		{ set_detector_map( coord_map ); }

	//! Reads map from file and stores it.
	void read_detectormap_from_file( const std::string& filename );
	//! stores the map
	/*! applies sorting to standard STIR order */
    void set_detector_map( const det_pos_to_coord_type& coord_map );

	stir::DetectionPosition<> get_det_pos_for_index(const stir::DetectionPosition<>& index) const
	{
		return input_index_to_det_pos.at(index);
    }
	//! Returns a cartesian coordinate given a detection position.
	stir::CartesianCoordinate3D<float> get_coordinate_for_det_pos( const stir::DetectionPosition<>& det_pos ) const
	{ 
		auto coord = det_pos_to_coord.at(det_pos);
		coord.x() += static_cast<float>(distribution(generator));
		coord.y() += static_cast<float>(distribution(generator));
		coord.z() += static_cast<float>(distribution(generator));
		return coord;
	}
	//! Returns a cartesian coordinate given an (unsorted) index.
	stir::CartesianCoordinate3D<float> get_coordinate_for_index( const stir::DetectionPosition<>& index ) const
	{
		return get_coordinate_for_det_pos(get_det_pos_for_index(index));
	}

        Succeeded
          find_detection_position_given_cartesian_coordinate(DetectionPosition<>& det_pos,
                                                             const CartesianCoordinate3D<float>& cart_coord) const;

        unsigned get_num_tangential_coords() const
	{ return num_tangential_coords; }
	unsigned get_num_axial_coords() const
	{ return num_axial_coords; }
	unsigned get_num_radial_coords() const
	{ return num_radial_coords; }

protected:

 DetectorCoordinateMap(double sigma = 0.0) :
        sigma(sigma),
        distribution(0.0, sigma)
  {}
private:
  unsigned num_tangential_coords;
  unsigned num_axial_coords;
  unsigned num_radial_coords;
  unordered_to_ordered_det_pos_type input_index_to_det_pos;
  det_pos_to_coord_type det_pos_to_coord;
  std::map<stir::CartesianCoordinate3D<float>,
    stir::DetectionPosition<>> detection_position_map_given_cartesian_coord_keys_3_decimal;
  std::map<stir::CartesianCoordinate3D<float>,
    stir::DetectionPosition<>> detection_position_map_given_cartesian_coord_keys_2_decimal;

  const double sigma;
  mutable std::default_random_engine generator;
  mutable std::normal_distribution<double> distribution;

  static det_pos_to_coord_type
    read_detectormap_from_file_help( const std::string& crystal_map_name );

};

END_NAMESPACE_STIR
#endif
