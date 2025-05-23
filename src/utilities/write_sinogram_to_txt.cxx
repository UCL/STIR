/* write_sinogram_to_txt.cxx

 Extracts sinogram to txt file
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

#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"

#include "stir/IO/interfile.h"
#include "stir/shared_ptr.h"

#include <fstream> 
#include <iostream>

USING_NAMESPACE_STIR
using std::cerr;
using std::cout;
using std::endl;
using std::ofstream;
int main( int argc, char** argv )
{
	if( argc < 4 )
	{
		cerr << "Usage: " << argv[0] << " filename.hs axial_position segment_number" << endl;
		return EXIT_FAILURE;
	}
	std::string filename(argv[1]);
	
	shared_ptr<ProjData> projdata_sptr = ProjData::read_from_file(filename.c_str());
	Sinogram<float> sino = projdata_sptr->get_sinogram(atoi(argv[2]), atoi(argv[3]) );
	
	std::string outfilename;
	size_t lastdot = filename.find_last_of(".");
	if( lastdot == std::string::npos ) outfilename = filename;
	else outfilename = filename.substr(0,lastdot);
	outfilename.append("_axialposition");
	outfilename.append(argv[2]);
	outfilename.append("_segment");
	outfilename.append(argv[3]);
	outfilename.append(".csv");
	
	ofstream outfile(outfilename.c_str());
	outfile << "#\tsegment=" << sino.get_segment_num() << "\taxial_pos=" << sino.get_axial_pos_num() << endl;
	
	for( int view = sino.get_min_view_num(); view <= sino.get_max_view_num(); view++ ) {
		for( int tpos = sino.get_min_tangential_pos_num(); tpos <= sino.get_max_tangential_pos_num(); tpos++ ){
			outfile << sino[view][tpos] << "\t";
		}
		outfile << endl;
	}

	outfile.close();
	return EXIT_SUCCESS;
} 
