/*!
  \file
  \ingroup utilities

  \brief A utility that creates a template projection data.
  \par Usage
  \verbatim
  create_projdata_template output_filename
  \endverbatim
  This will ask questions to the user about the scanner, the data size,
  etc. It will then output new projection data (in Interfile format).
  However, the binary file will not contain any data.

  This utility is mainly useful to create a template that can then
  be used for other STIR utilities (such as fwdtest, lm_to_projdata etc.).

  \author Kris Thielemans
*/
/*
    Copyright (C) 2004, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London    
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInterfile.h"
#include "stir/ImagingModality.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfo.h"

using std::cerr;


USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
  
  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " output_filename\n";
    return EXIT_FAILURE;
  }


  shared_ptr<ProjDataInfo> proj_data_info_sptr(ProjDataInfo::ask_parameters());
  
  const std::string output_file_name = argv[1];
  shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo);
  // TODO, Currently all stir::Scanner types are PET.
  exam_info_sptr->imaging_modality = ImagingModality::PT;
  // If TOF activated -- No mashing factor will produce surrealistic sinograms
  //if ( proj_data_info_sptr->get_num_tof_poss() >1)
    //  shared_ptr<ProjData> proj_data_sptr(new ProjDataInterfile(exam_info_sptr, proj_data_info_sptr, output_file_name, std::ios::out,
      //                                                          ProjDataFromStream::Timing_Segment_View_AxialPos_TangPos));
  //else
      shared_ptr<ProjData> proj_data_sptr(new ProjDataInterfile(exam_info_sptr, proj_data_info_sptr, output_file_name));

  return EXIT_SUCCESS;
}
