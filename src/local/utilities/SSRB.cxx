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
  if (argc > 6 || argc < 4 )
    {
      cerr << "Usage:\n"
	   << argv[0] << " output_filename input_projdata_name span [ do_norm [max_in_segment_num_to_process ]]\n"
	   << "do_norm has to be 1 (normalise the result) or 0\n"
	   << "max_in_segment_num_to_process defaults to all segments\n";
      exit(EXIT_FAILURE);
    }
  const string  output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);
  const int span = atoi(argv[3]);

  const bool do_norm = argc<=4 ? true : atoi(argv[4]) != 0;
  const int max_segment_num_to_process = argc <=5 ? -1 : atoi(argv[5]);

  SSRB(output_filename,
       *in_projdata_ptr,
       span,
       max_segment_num_to_process,
       do_norm
       );
  return EXIT_SUCCESS;
}
