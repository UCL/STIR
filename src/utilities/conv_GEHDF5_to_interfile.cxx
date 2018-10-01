/*
 Copyright (C) 2005- 2009, Hammersmith Imanet Ltd
 Copyright (C) 2010- 2013, King's College London
 Copyright (C) 2013, University College London
 Copyright (C) 2018, University of Leeds
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
 \brief  This program converts GE HDF5 sinogram into STIR interfile format

 \author Palak Wadhwa
 \author Nikos Efthimiou
 */

#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Sinogram.h"
#include "stir/IO/read_data.h"
#include "stir/Succeeded.h"
#include "stir/NumericType.h"
#include "stir/ProjDataFromHDF5.h"

#define NUMARG 4

int main(int argc,char **argv)
{
    using namespace stir;

    static const char * const options[]={
        "argv[1]  output_filename\n"
        "argv[2]  RDF_filename\n"
        "argv[3]  template_projdata"
    };
    if (argc!=NUMARG){
        std::cerr << "\n\nConvert GE HDF5 sinogram to STIR\n\n";
        std::cerr << "Not enough arguments !!! ..\n";
        for (int i=1;i<NUMARG;i++) std::cerr << options[i-1];
        exit(EXIT_FAILURE);
    }

    const std::string output_filename(argv[1]);
    const std::string rdf_filename(argv[2]);
    shared_ptr<ProjDataInfo> template_projdata_info_sptr =
            ProjData::read_from_file(argv[3])->get_proj_data_info_sptr();

//    ProjDataInterfile proj_data(template_projdata_ptr->get_exam_info_sptr(), template_projdata_ptr->get_proj_data_info_sptr(),
//                                output_filename, std::ios::out);

    ProjDataFromHDF5 projDataGE(template_projdata_info_sptr, rdf_filename);

   //for (int i_seg = projDataGE.get_min_segment_num(); i_seg <= projDataGE.get_max_segment_num(); ++i_seg)
     //   for(int i_view = projDataGE.get_min_view_num(); i_view <= projDataGE.get_max_view_num(); ++i_view)
       // {
    int i_view = 0;
    int i_seg = 1;
    Viewgram<float> ret_viewgram = projDataGE.get_viewgram(i_view, i_seg);

           // for(int i_axial = ret_viewgram.get_min_axial_pos_num(); i_axial <= ret_viewgram.get_max_axial_pos_num(); ++i_axial)
            //{
              //  std::cout << std::endl;
              //  for (int i_tang = ret_viewgram.get_min_tangential_pos_num(); i_tang <= ret_viewgram.get_max_tangential_pos_num(); ++i_tang)
              //  {
              //      std::cout << ret_viewgram.at(i_axial).at(i_tang) << " ";
              //  }
           // }
      //  }

//    for (int segment_num= template_projdata_ptr->get_min_segment_num(); segment_num <= template_projdata_ptr->get_max_segment_num(); segment_num++) {
//        Viewgram<float> view = projDataGE.get_viewgram(1, segment_num, false);
//        proj_data.set_viewgram(view);
//    }

    return EXIT_SUCCESS;
}
