//
// $Id: 
//

/*!
\file
\ingroup utilities
\brief process sinogram data

\author Sanida Mustafovic 
\author Kris Thielemans 
\author PARAPET project

$Date: 
$Revision: 

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/



#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h" 
#include "stir/recon_array_functions.h"
#include "stir/display.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/shared_ptr.h"

#include <numeric>
#include <fstream> 
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
#endif



USING_NAMESPACE_STIR

int 
main(int argc, char **argv)

{
  ProjDataInfo* new_data_info_ptr;
  shared_ptr<ProjData> proj_data_ptr_first;
  shared_ptr<ProjData> proj_data_ptr_second;
  
  
  if (argc!=4)
  {
    cerr << " USAGE: add_projections: output projdata filename,first projdata filename, second projdata filename " << endl;
  }
  else
  { 
    proj_data_ptr_first = ProjData::read_from_file(argv[2]); 
    proj_data_ptr_second = ProjData::read_from_file(argv[3]); 
  }
  
  const string output_file_name = argv[1];
  const ProjDataInfo * proj_data_info_ptr = 
    proj_data_ptr_first->get_proj_data_info_ptr();

  new_data_info_ptr = proj_data_info_ptr->clone();
  shared_ptr<iostream> sino_stream = new fstream (output_file_name.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("fwdtest: error opening file %s\n",output_file_name.c_str());
  }

  shared_ptr<ProjDataFromStream> proj_data_ptr =
    new ProjDataFromStream(new_data_info_ptr,sino_stream);
   
  for (int segment_num = proj_data_ptr_first->get_min_segment_num();segment_num <=proj_data_ptr_first->get_max_segment_num();segment_num++)
  {   
    SegmentByView<float> segment_by_view_1  = 
      proj_data_ptr_first->get_segment_by_view(segment_num);
    SegmentByView<float> segment_by_view_2  = 
      proj_data_ptr_second->get_segment_by_view(segment_num);

    
    segment_by_view_1 +=segment_by_view_2;
    if (!(proj_data_ptr->set_segment(segment_by_view_1) == Succeeded::yes))
      warning("Error set_segment %d\n", segment_num);   
  }
  
  
  write_basic_interfile_PDFS_header(output_file_name, *proj_data_ptr);

  return EXIT_SUCCESS;
}
