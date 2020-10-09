//
//
/*
    Copyright (C) 2002- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup utilities
  \brief Main program for stir::SSRB

  \author Kris Thielemans


   \par Usage:
   \code
   SSRB [-t num_tangential_poss_to_trim] \
        output_filename input_projdata_name [num_segments_to_combine \
      [ num_views_to_combine [do_normalisation [max_in_segment_num_to_process ]]]]
   \endcode
   \param num_segments_to_combine has to be odd. It is used as the number of segments
      in the original data to combine.
   \param num_views_to_combine has to be at least 1 (which is the default). 
      It is used as the number of views in the original data to combine.
   \param num_tangential_poss_to_trim has to be smaller than the available number
      of tangential positions.
   \param do_normalisation has to be 1 (normalise the result, which is the default) or 0
   \param max_in_segment_num_to_process defaults to all segments

  \par Example:
  \code
  SSRB out in.hs 3 2
  \endcode
  If in.hs is a file without axial compression (span=1) nor mashing, then the
  output would correspond to a span=3 file with mashing factor 2, and would be
  normalised (at least as far as SSRB concerns). \a num_segments_to_combine=3
  results in ring differences -1,0,1 to be combined all into segment 0, etc. 
  \see 
  stir::SSRB(const std::string& output_filename,
             const stir::ProjData& in_projdata,
	     const int num_segments_to_combine,
	     const int num_views_to_combine,
	     const int num_tang_poss_to_trim,
	     const bool do_normalisation,
	     const int max_in_segment_num_to_process
     )
  for info on parameters and restrictions.
*/
#include "stir/ProjData.h"
#include "stir/shared_ptr.h"
#include "stir/SSRB.h"
#include <string>
#include <string.h>
#include "stir/ProjDataInterfile.h"

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc > 7 || argc < 3 )
    {
      cerr << "Usage:\n"
	   << argv[0] << " [-t num_tangential_poss_to_trim] \\\n"
	   << "\toutput_filename input_projdata_name \\\n"
           << "\t[num_segments_to_combine \\\n"
           <<"\t[ num_views_to_combine [do_norm [max_in_segment_num_to_process ]]]]\n"
	   << "num_segments_to_combine has to be odd. It is used as the number of segments\n"
	   << "  in the original data to combine.\n"
           << "num_views_to_combine has to be at least 1 (which is the default)\n"
	   << "num_tangential_poss_to_trim has to be smaller than the available number\n"
	   << "  of tangential positions.\n"
	   << "do_norm has to be 1 (normalise the result, which is the default) or 0\n"
	   << "max_in_segment_num_to_process defaults to all segments\n";
      exit(EXIT_FAILURE);
    }

  // Define defaults
  int span = 1;
  int num_views_to_combine = 1;
  int max_segment_num_to_process = -1;
  bool do_norm = true;
  int num_tangential_poss_to_trim = 0;

  const string output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);

  if (strcmp(argv[3], "--template")==0)
  {
    //3rd argument is "--template", load this template sinogram
    shared_ptr<ProjData> template_projdata_ptr = ProjData::read_from_file(argv[4]);
    ProjDataInterfile out_proj_data(in_projdata_ptr->get_exam_info_sptr(),
                                    template_projdata_ptr->get_proj_data_info_sptr(), output_filename, std::ios::out);
    if (argc >5)
      if (atoi(argv[5]) == 0)
        do_norm = false;
    SSRB(out_proj_data, *in_projdata_ptr, do_norm);

  } else {
    // All this needs checking for standard usage
    if (argc>1 && strcmp(argv[1], "-t")==0)
    {
      num_tangential_poss_to_trim = atoi(argv[2]);
      argc -= 2; argv += 2;
    }
    if (argc >3)
      span = atoi(argv[3]);
    if (argc >4)
      num_views_to_combine = atoi(argv[4]);
    if (argc >5)
      if (atoi(argv[5]) == 0)
        do_norm = false;
    if (argc >6)
      max_segment_num_to_process = -atoi(argv[6]);

    // do standard SSRB
    SSRB(output_filename,
         *in_projdata_ptr,
         span,
         num_views_to_combine,
         num_tangential_poss_to_trim,
         do_norm,
         max_segment_num_to_process
    );
  }


  return EXIT_SUCCESS;
}
