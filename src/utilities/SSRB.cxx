//
//
/*
    Copyright (C) 2002- 2005, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup utilities
  \brief Main program for stir::SSRB

  \author Kris Thielemans & Robert Twyman


   \par Usage:
   \code
   SSRB [-t num_tangential_poss_to_trim] \
        output_filename input_projdata_name [num_segments_to_combine \
      [ num_views_to_combine [do_normalisation [max_in_segment_num_to_process ]]]]
   \endcode
   \code
   SSRB --template template_projdata_filename output_filename input_projdata_name [do_normalisation]
   \endcode
   \param num_segments_to_combine has to be odd. It is used as the number of segments
      in the original data to combine.
   \param num_views_to_combine has to be at least 1 (which is the default). 
      It is used as the number of views in the original data to combine.
   \param num_tangential_poss_to_trim has to be smaller than the available number
      of tangential positions.
   \param do_normalisation has to be 1 (normalise the result, which is the default) or 0
   \param max_in_segment_num_to_process defaults to all segments
   \param template_projdata_filename , indicated by the --template option, a template projection
      data which SSRB will map to.

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
  \code
  SSRB --template template_sino.hs out in.hs 1
  \endcode
  This use SSRB to convert in.hs into the shape of template_sino.hs and save the result as interile out.
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


static void print_usage_and_exit(const std::string& prog_name)
{
  cerr << "Usage:\n\n"
       << "Two options:\n"
       << "\t - Classical STIR SSRB method\n"
       << prog_name << " [-t num_tangential_poss_to_trim] \\\n"
       << "\toutput_filename input_projdata_name \\\n"
       << "\t[num_segments_to_combine \\\n"
       <<"\t[ num_views_to_combine [do_norm [max_in_segment_num_to_process ]]]]\n"
       << "num_segments_to_combine has to be odd. It is used as the number of segments\n"
       << "  in the original data to combine.\n"
       << "num_views_to_combine has to be at least 1 (which is the default)\n"
       << "num_tangential_poss_to_trim has to be smaller than the available number\n"
       << "  of tangential positions.\n"
       << "do_norm has to be 1 (normalise the result, which is the default) or 0\n"
       << "max_in_segment_num_to_process defaults to all segments\n"
       << "\n\t - Template Based SSRB method\n"
       << prog_name << "--template template_projdata_filename output_filename input_projdata_name [do_norm]\n"
       << "template_projdata_filename the format of the output sinogram.\n"
       << "output_filename, input_projdata_name, and do_norm are as above.\n"
       << "SSRB act in the same way as Classical but allows for even num_segments_to_combine.\n";
  exit(EXIT_FAILURE);
}

void classic_SSRB(int argc, char **argv)
{
  // Does the standard method of SSRB based upon numerical inputs
  int num_tangential_poss_to_trim = 0;
  if (argc>1 && strcmp(argv[1], "-t")==0)
  {
    num_tangential_poss_to_trim = atoi(argv[2]);
    argc -= 2; argv += 2;
  }
  const string output_filename = argv[1];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[2]);
  const int num_segments_to_combine = argc <= 3 ? 1 : atoi(argv[3]);
  const int num_views_to_combine = argc<=4 ? 1 : atoi(argv[4]);
  const bool do_norm = argc<=5 ? true : atoi(argv[5]) != 0;
  const int max_segment_num_to_process = argc <=6 ? -1 : atoi(argv[6]);
  // do standard SSRB
  SSRB(output_filename,
       *in_projdata_ptr,
       num_segments_to_combine,
       num_views_to_combine,
       num_tangential_poss_to_trim,
       do_norm,
       max_segment_num_to_process
  );
}

void template_based_SSRB(int argc, char **argv)
{
  // Does the template based SSRB whereby the target template is given as an argument
  shared_ptr<ProjData> template_projdata_ptr = ProjData::read_from_file(argv[2]);
  const string output_filename = argv[3];
  shared_ptr<ProjData> in_projdata_ptr = ProjData::read_from_file(argv[4]);
  ProjDataInterfile out_proj_data(in_projdata_ptr->get_exam_info_sptr(),
                                  template_projdata_ptr->get_proj_data_info_sptr(), output_filename, std::ios::out);
  const bool do_norm = argc<=5 ? true : atoi(argv[5]) != 0;
  SSRB(out_proj_data, *in_projdata_ptr, do_norm);
}

int main(int argc, char **argv)
{
  if (argc > 7 || argc < 3 )
  {
    print_usage_and_exit(argv[0]);
  }

  if (strcmp(argv[1], "--template")==0)
  {
    template_based_SSRB(argc, argv);
  } else {
    classic_SSRB(argc, argv);
  }

  return EXIT_SUCCESS;
}
