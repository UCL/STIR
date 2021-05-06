/*!

  \file
  \ingroup utilities
  \ingroup GE
  \brief Construct randoms as a product of singles estimates

  Dead-time is not taken into account.

  \todo We currently assume F-18 for decay.

  \author Palak Wadhwa
  \author Kris Thielemans

*/
/*
  Copyright (C) 2017- 2019, University of Leeds
  Copyright (C) 2020, 2021, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/IndexRange2D.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/info.h"
#include <iostream>
#include <string>
#include "stir/data/SinglesRatesFromGEHDF5.h"
#include "stir/data/randoms_from_singles.h"
#include <boost/format.hpp>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::string;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc!=4 && argc!=3)
    {
      cerr << "Usage: " << argv[0]
           << " out_filename GE_RDF_filename [template_projdata]\n"
           << "The template is used for size- and time-frame-info, but actual counts are ignored.\n"
           << "If no template is specified, we will use the normal GE sizes and the time frame information of the RDF.\n";
      return EXIT_FAILURE;
    }
  
  const string input_filename = argv[2];
  const string output_file_name = argv[1];
  const string program_name = argv[0];
  shared_ptr<const ProjDataInfo> proj_data_info_sptr;
  shared_ptr<const ExamInfo> exam_info_sptr;

  GE::RDF_HDF5::GEHDF5Wrapper input_file(input_filename);
  std::string template_filename;
  if (argc==4)
    {
      template_filename = argv[3];
      shared_ptr<ProjData> template_projdata_sptr = ProjData::read_from_file(template_filename);
      proj_data_info_sptr = template_projdata_sptr->get_proj_data_info_sptr();
      exam_info_sptr = template_projdata_sptr->get_exam_info_sptr();
    }
  else
    {
      template_filename = input_filename;
      proj_data_info_sptr = input_file.get_proj_data_info_sptr();
      exam_info_sptr = input_file.get_exam_info_sptr();
    }

  if (exam_info_sptr->get_time_frame_definitions().get_num_time_frames()==0 ||
      exam_info_sptr->get_time_frame_definitions().get_duration(1) < .0001)
    error("Missing time-frame information in \"" + template_filename +'\"');

  ProjDataInterfile 
    proj_data(exam_info_sptr,
              proj_data_info_sptr->create_shared_clone(),
              output_file_name);

  GE::RDF_HDF5::SinglesRatesFromGEHDF5  singles(input_filename);
  const float coincidence_time_window = input_file.get_coincidence_time_window();

  randoms_from_singles(proj_data, singles, coincidence_time_window);
  return EXIT_SUCCESS;
}
