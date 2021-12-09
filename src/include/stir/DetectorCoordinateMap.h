/*
	Copyright 2015 ETH Zurich, Institute of Particle Physics
	Copyright 2020 Positrigo AG, Zurich

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
 */

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::DetectorCoordinateMap

  \author Jannis Fischer
*/

#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>

#include "stir/CartesianCoordinate3D.h"
#include "stir/DetectionPosition.h"

#ifndef __stir_DetectorCoordinateMap_H__
#define __stir_DetectorCoordinateMap_H__

START_NAMESPACE_STIR

/*! Class providing map functionality to convert detector indices to spatial coordinates. 
	Map files can have 5 or 6 tab- or comma-separated columns. Lines beginning with '#' are ignored. The layer column is optional
	\par Format:
	ring,detector,(layer,)x,y,z
	An empty line will terminate the reading at that line.
*/
class DetectorCoordinateMap
{
public:
	//! Constructor calls read_detectormap_from_file( filename ).
	DetectorCoordinateMap(const std::string& filename, double sigma = 0.0) :
		sigma(sigma),
		distribution(0.0, sigma)
		{ read_detectormap_from_file( filename ); }

	//! Reads map from file and stores it.
	void read_detectormap_from_file( const std::string& filename );

	//! Returns a cartesian coordinate given a detection position.
	stir::CartesianCoordinate3D<float> get_detector_coordinate( const stir::DetectionPosition<>& det_pos )
	{ 
		auto coord = coord_map.at(det_pos); 
		coord.x() += distribution(generator);
		coord.y() += distribution(generator);
		coord.z() += distribution(generator);
		return coord;
	}

private:
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

	boost::unordered_map< stir::DetectionPosition<>, stir::CartesianCoordinate3D<float>, ihash > coord_map;
	const double sigma;
	std::default_random_engine generator;
	std::normal_distribution<double> distribution;
};

END_NAMESPACE_STIR
#endif
