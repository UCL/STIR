//
// $Id: 
//

/*!
\file
\ingroup local/utilities
\brief modify_proj_data 

\author Sanida Mustafovic 
*/

#include "stir/ProjDataFromStream.h"
#include "stir/SegmentByView.h"
#include "stir/interfile.h"
#include "stir/utilities.h"
#include "stir/Succeeded.h"


#include <fstream> 
#include <iostream> 

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ofstream;
using std::cout;
#endif



USING_NAMESPACE_STIR

int 
main(int argc, char **argv)
{
  
  if(argc!=3)
  {
    cerr<< "Usage: " << argv[0] << " in_projdata out_projdata \n";
    exit(EXIT_FAILURE);
  }
  
  shared_ptr<ProjData> proj_data = 0;
  proj_data =  ProjData::read_from_file(argv[1]); 
  
  const ProjDataInfo * proj_data_info_ptr = 
    (*proj_data).get_proj_data_info_ptr();

  shared_ptr<iostream> sino_stream = new fstream (argv[2], ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("modify_proj_data: error opening file %s\n",argv[2]);
  }
 
  ProjDataInfo* new_proj_data_info_ptr= proj_data_info_ptr->clone();
  new_proj_data_info_ptr->set_min_tangential_pos_num(-64);
  new_proj_data_info_ptr->set_max_tangential_pos_num(64); 
  
  shared_ptr<ProjDataFromStream> proj_data_ptr_out =
    new ProjDataFromStream(new_proj_data_info_ptr,sino_stream);

  
  int min_segment_num = proj_data_info_ptr->get_min_segment_num();
  int max_segment_num = proj_data_info_ptr->get_max_segment_num();
     

  for (int segment_num = min_segment_num; segment_num<= max_segment_num;
  segment_num++) 
    for ( int view_num = proj_data_info_ptr->get_min_view_num();
    view_num<=proj_data_info_ptr->get_max_view_num(); view_num++)
    {
      Viewgram<float> viewgram_in = proj_data->get_viewgram(view_num,segment_num);
      Viewgram<float> viewgram_out = new_proj_data_info_ptr->get_empty_viewgram(view_num,segment_num);

      for (int i= viewgram_in.get_min_axial_pos_num(); i<=viewgram_in.get_max_axial_pos_num();i++)
	for (int j= -64; j<=64;j++)
	{
	  viewgram_out[i][j] = viewgram_in[i][j];
	}
	proj_data_ptr_out->set_viewgram(viewgram_out);

    }

   write_basic_interfile_PDFS_header(argv[2], *proj_data_ptr_out);

   return EXIT_SUCCESS;
}
