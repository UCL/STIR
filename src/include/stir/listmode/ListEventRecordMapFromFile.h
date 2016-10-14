/* ListEventRecordMapFromFile.h
 Read List-Mode Event Data using map from file: Header File
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

/*!

  \file
  \ingroup listmode
  \brief Declaration of class stir::CListEventRecordMapFromFile

  \author Jannis Fischer
*/

#include <fstream>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/unordered_map.hpp>

#include "stir/CartesianCoordinate3D.h"
#include "stir/DetectionPosition.h"

#ifndef __local_SAFIR_ListEventRecordMapFromFile_H__
#define __local_SAFIR_ListEventRecordMapFromFile_H__

START_NAMESPACE_STIR

//! Class providing map functionality to convert detector indices to spatial coordinates. 
class ListEventRecordMapFromFile
{
public:
	//! Constructor calls read_detectormap_from_file( filename ).
	ListEventRecordMapFromFile(const std::string& filename)
		{ read_detectormap_from_file( filename ); }

	/*! Reads map from file and stores it.
	Map files can have 5 or 6 tab- or comma-separated columns. Lines beginning with '#' are ignored. The layer column is optional
	\par Format:
	ring,detector,(layer,)x,y,z
	An empty line will terminate the reading at that line.
	*/
	void read_detectormap_from_file( const std::string& filename );

	//! Returns a cartesian coordinate given a detection position.
	stir::CartesianCoordinate3D<float> get_detector_coordinate( stir::DetectionPosition<>* det_pos )
		{ return coord_map.at(*det_pos); }

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
};

END_NAMESPACE_STIR
#endif
