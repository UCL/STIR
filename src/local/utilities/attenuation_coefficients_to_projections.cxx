//
// $Id$
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


#include "stir/ProjData.h"
#include "stir/ProjDataInterfile.h"
#include "stir/Viewgram.h"
#include "stir/ArrayFunction.h"

#include <iostream>

#ifndef STIR_NO_NAMESPACES
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

  shared_ptr<ProjData> out_proj_data_ptr =
    new ProjDataInterfile(attenuation_proj_data_ptr->get_proj_data_info_ptr()->clone(),
			  output_file_name);

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

