//
// $Id:
//

/*!
  \file
  \ingroup utilities

  \brief Takes the logarithm of attenuation coefficients to 'convert'
  them to line integrals.
  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Year:$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/interfile.h"
#include "stir/ProjData.h"
#include "stir/ProjDataFromStream.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"

#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::iostream;
using std::endl;
using std::cerr;
using std::endl;
#endif


USING_NAMESPACE_STIR

int 
main (int argc, char * argv[])
{
  
  if (argc!=3)
  {
    cerr<<"\nUsage: " <<argv[0] << " <output filename > <input proj_data file name>  \n"<<endl;
    return EXIT_FAILURE;
  }
  
  shared_ptr <ProjData> attenuation_proj_data_ptr =
    ProjData::read_from_file(argv[2]);

  const string output_file_name = argv[1];
  shared_ptr<iostream> sino_stream = new fstream (output_file_name.c_str(), ios::out|ios::binary);
  if (!sino_stream->good())
  {
    error("%s: error opening file %s\n",argv[0],output_file_name.c_str());
  }

  
   shared_ptr<ProjDataFromStream> out_proj_data_ptr =
    new ProjDataFromStream(attenuation_proj_data_ptr->get_proj_data_info_ptr()->clone(),sino_stream);
  
  write_basic_interfile_PDFS_header(output_file_name, *out_proj_data_ptr);

  for (int segment_num = attenuation_proj_data_ptr->get_min_segment_num(); 
       segment_num<= attenuation_proj_data_ptr->get_max_segment_num();
       ++segment_num)
    for ( int view_num = attenuation_proj_data_ptr->get_min_view_num();
	  view_num<=attenuation_proj_data_ptr->get_max_view_num(); 
	  ++view_num)
    {
      Viewgram<float> viewgram = attenuation_proj_data_ptr->get_viewgram(view_num,segment_num);
      in_place_log(viewgram);
      out_proj_data_ptr->set_viewgram(viewgram);
    }
    
  
  return EXIT_SUCCESS;
}

