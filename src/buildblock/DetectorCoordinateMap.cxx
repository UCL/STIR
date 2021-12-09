/* 
	Copyright 2015 ETH Zurich, Institute of Particle Physics

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock
  \brief Implementation of class stir::DetectorCoordinateMap

  \author Jannis Fischer
*/

#include "stir/error.h"
#include "stir/DetectorCoordinateMap.h"

START_NAMESPACE_STIR
	
void DetectorCoordinateMap::read_detectormap_from_file( const std::string& filename )
{
	std::ifstream myfile(filename.c_str());
	if( !myfile ) 
	{
		error("Error opening file " + filename + ".\n");
		return;
	}

//	char line[80];
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
}
END_NAMESPACE_STIR
