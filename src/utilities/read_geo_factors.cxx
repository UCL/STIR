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
 \brief  This program reads geometric correction factors from the HDF5 file and saves it
 as STIR interfile.
 \author Palak Wadhwa
 */

#include "stir/ProjDataInterfile.h"
#include "stir/ExamInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataFromHDF5.h"
#include "stir/IO/read_data.h"
#include "stir/Succeeded.h"
#include "stir/NumericType.h"
#include "stir/IO/HDF5Wrapper.h"
#include "stir/IndexRange3D.h"
#include "stir/DetectionPosition.h"
#include "stir/DetectionPositionPair.h"
#include "stir/shared_ptr.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange.h"
#include "stir/Bin.h"
#include "stir/display.h"
#include "stir/IO/read_data.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/ByteOrder.h"
#include "stir/is_null_ptr.h"
#include "stir/modulo.h"
#include <algorithm>
#include <fstream>
#include <cctype>
#ifndef STIR_NO_NAMESPACES
using std::ofstream;
using std::fstream;
using std::ios;
#endif

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
        std::cerr << "\n\nConvert geometric correction factors into STIR interfile.\n\n";
        std::cerr << "Not enough arguments !!! ..\n";
        for (int i=1;i<NUMARG;i++) std::cerr << options[i-1];
        exit(EXIT_FAILURE);
    }

    const std::string output_filename(argv[1]);
    const std::string rdf_filename(argv[2]);
    shared_ptr<ProjDataInfo> template_projdata_info_sptr =
            ProjData::read_from_file(argv[3])->get_proj_data_info_sptr();

    shared_ptr<ProjData> template_projdata_ptr =
            ProjData::read_from_file(argv[3]);

    ProjDataInterfile proj_data(template_projdata_ptr->get_exam_info_sptr(), template_projdata_ptr->get_proj_data_info_sptr(),
                                output_filename, std::ios::out);

    ProjDataFromHDF5 projDataGE(template_projdata_info_sptr, rdf_filename);
  //  for (int i_seg = projDataGE.get_min_segment_num(); i_seg <= projDataGE.get_max_segment_num(); ++i_seg)
    //        for(int i_view = projDataGE.get_min_view_num(); i_view <= projDataGE.get_max_view_num(); ++i_view)
      //      {
    // PW For initial testing segment and view number are assumed to be 0. The commented piece of code
    // will be used after the testing.

    int i_seg = 0;
    int i_view = 0;
    // PW Viewgram and geometric correction factors are initialised.
                 Viewgram<float> ret_viewgram = proj_data.get_empty_viewgram(i_view,i_seg);
                 ret_viewgram.fill(0.0);
                 Array<2,float> geometric_factors;
                 geometric_factors = Array<2,float>(IndexRange2D(1981,357));
//               geometric_factors = Array<2,float>(IndexRange2D(projDataGE.get_num_tangential_poss(),static_cast<unsigned long long int>(projDataGE.get_num_axial_poss(i_seg))));
                 std::cout<<"This is the number of axial positions "<<static_cast<unsigned long long int>(projDataGE.get_num_axial_poss(i_seg))<<std::endl;
                 std::cout<<"This is the axial offset "<<projDataGE.seg_ax_offset[projDataGE.find_segment_index_in_sequence(i_seg)]<<std::endl;
                 std::cout<<projDataGE.get_num_tangential_poss()<<std::endl;

                 // PW HDF5 Wrapper is initialised here and the address of the data is read subsequently.
                 shared_ptr<HDF5Wrapper> m_input_hdf5_sptr;
                 m_input_hdf5_sptr.reset(new HDF5Wrapper(rdf_filename));
                 m_input_hdf5_sptr->initialise_geo_factors_data("",modulo(i_view,16));

                 // PW Here the data is read from the HDF5 array.
                 std::array<unsigned long long int, 2> stride = {1, 1};
                // std::array<unsigned long long int, 2> count = {static_cast<unsigned long long int>(projDataGE.get_num_axial_poss(i_seg)),
                 // static_cast<unsigned long long int>(projDataGE.get_num_tangential_poss())};

                 std::array<unsigned long long int, 2> count = {1981,357};
                 std::array<unsigned long long int, 2> offset = {0, 0};
                 std::array<unsigned long long int, 2> block = {1, 1};
                 unsigned int total_size = projDataGE.get_num_tangential_poss() * 1981;

             //    unsigned int total_size = projDataGE.get_num_tangential_poss() * static_cast<unsigned long long int>(projDataGE.get_num_axial_poss(i_seg));
                 stir::Array<1, unsigned char> tmp(0, total_size-1);

                 m_input_hdf5_sptr->get_from_2d_dataset(offset, count, stride, block, tmp);
// PW The tmp array data is copied into 2D array of geometric correction factors of size 357x1981.
                 std::copy(tmp.begin(),tmp.end(),geometric_factors.begin_all());

                //for (int tang_pos = ret_viewgram.get_min_tangential_pos_num(), i_tang = 0; tang_pos <= ret_viewgram.get_max_tangential_pos_num(),
                  //   i_tang<=static_cast<unsigned long long int>(projDataGE.get_num_tangential_poss())-1; ++tang_pos, ++i_tang)
                // for(int i_axial=0, axial_pos = projDataGE.seg_ax_offset[projDataGE.find_segment_index_in_sequence(i_seg)]; i_axial<=static_cast<unsigned long long int>(projDataGE.get_num_axial_poss(i_seg))-1 ,
                   //  axial_pos <= projDataGE.seg_ax_offset[projDataGE.find_segment_index_in_sequence(i_seg)]+static_cast<unsigned long long int>(projDataGE.get_num_axial_poss(i_seg))-1; i_axial++, axial_pos++)
                   // {
                   //      ret_viewgram[i_axial][tang_pos] = geometric_factors[i_tang][axial_pos];
                   // }

// The geometric correction factors data from first 89 positions i.e. segment 0 are copied into ret_viewgram.
                     for (int tang_pos = ret_viewgram.get_min_tangential_pos_num(), i_tang = 0;
                                             tang_pos <= ret_viewgram.get_max_tangential_pos_num(), i_tang<=356;
                                             ++tang_pos, ++i_tang)
                                            for(int i_axial=0, axial_pos = 0;
                                                i_axial<=88, axial_pos <= 88;
                                                i_axial++, axial_pos++)
                                            {
                                                ret_viewgram[i_axial][tang_pos] = geometric_factors[axial_pos][i_tang];
                                            }
                   // This is now saved in STIR viewgrams.
                     proj_data.set_viewgram(ret_viewgram);

                  //  ofstream write_geo;
                // write_geo.open("uncompressed_buffer_data.txt",ios::out);
                // for ( int i =buffer.get_min_index(); i<=buffer.get_max_index();i++)
                  // {

                    //  write_geo << buffer[i] << "   " ;
                      //  }
                        //       write_geo << std::endl;

                 //   }
    return EXIT_SUCCESS;
}
