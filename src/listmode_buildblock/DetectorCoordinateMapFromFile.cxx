/* DetectorCoordinateMapFromFile.cxx

 Read List-Mode Event Data using map from file:  Implementation
 Jannis Fischer
 jannis.fischer@cern.ch

	Copyright 2015 ETH Zurich, Institute of Particle Physics

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

#include "stir/error.h"
#include "stir/listmode/DetectorCoordinateMapFromFile.h"

START_NAMESPACE_STIR
	
void DetectorCoordinateMapFromFile::read_detectormap_from_file( const std::string& filename )
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
		coord[1] = atof(col[4+has_layer_index].c_str() );
		coord[2] = atof(col[3+has_layer_index].c_str() );
		coord[3] = atof(col[2+has_layer_index].c_str() );
		
		if( !has_layer_index ) detpos.radial_coord() = 0;
		else detpos.radial_coord() = atoi(col[2].c_str());
		detpos.axial_coord() = atoi(col[0].c_str());
		detpos.tangential_coord() = atoi(col[1].c_str());

		coord_map[detpos] = coord;
	}
}
END_NAMESPACE_STIR
