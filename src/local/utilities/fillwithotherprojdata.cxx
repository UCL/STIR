//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief A utility that just fills the projection data with input from somewhere else. Only useful when the first file is an a different file format (i.e. ECAT 7)

  \author Kris Thielemans

  $Date$
  $Revision$
*/


#include "utilities.h"
#include "interfile.h"
#include "ProjDataFromStream.h"
#include "SegmentByView.h"
#include "tomo/Succeeded.h"

#include <iostream> 
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ifstream;
using std::cout;
#endif


USING_NAMESPACE_TOMO

int main(int argc, char *argv[])
{ 
  
  if(argc!=3) 
  {
    cerr<<"Usage: " << argv[0] << " output_projdata_file input_projdata_file\n"
	<<"The output_projdata_file must exist already, and will be overwritten.\n"
       	<< endl; 
  }

  shared_ptr<ProjData> out_projdata_ptr = 
    ProjData::read_from_file(argv[1], ios::in|ios::out);
  shared_ptr<ProjData> in_projdata_ptr = 
    ProjData::read_from_file(argv[2]);
  
  if (*out_projdata_ptr->get_proj_data_info_ptr() !=
      *in_projdata_ptr->get_proj_data_info_ptr())
    {
      error("Projection data infos are incompatible\n");
    }

  for (int segment_num=out_projdata_ptr->get_min_segment_num();
       segment_num<=out_projdata_ptr->get_max_segment_num();
       ++segment_num)
    {
      SegmentByView<float> segment = in_projdata_ptr->get_segment_by_view(segment_num);
      out_projdata_ptr->set_segment(segment);
      
    }
  return EXIT_SUCCESS;

}
