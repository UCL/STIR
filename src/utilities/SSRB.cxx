//
// $Id$
//
/*!

  \file
  \ingroup utilities
  \brief Main program for SSRB

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "local/stir/SSRB.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc > 7 || argc < 4 )
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name extra_span [ extra_mash [do_norm [max_in_segment_num_to_process ]]]\n"
	   << "extra_span has to be odd. It is used as the number of segments\n"
	   << "  in the original data to combine.\n"
	   << "do_norm has to be 1 (normalise the result) or 0\n"
	   << "max_in_segment_num_to_process defaults to all segments\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);
  const int span = atoi(argv[3]);
  const int num_views_to_combine = argc<=4 ? 1 : atoi(argv[4]);
  const bool do_norm = argc<=5 ? true : atoi(argv[5]) != 0;
  const int max_segment_num_to_process = argc <=6 ? -1 : atoi(argv[5]);
  if (max_segment_num_to_process>0 && max_segment_num_to_process%2==0)
    {
      warning("SSRB: the 'extra_span' argument has to be odd, but it is %d. Aborting\n", 
	      max_segment_num_to_process);
      return EXIT_FAILURE;
    }
  SSRB(output_filename,
       *in_projdata_ptr,
       span,
       num_views_to_combine,
       do_norm,
       max_segment_num_to_process
       );
  return EXIT_SUCCESS;
}
