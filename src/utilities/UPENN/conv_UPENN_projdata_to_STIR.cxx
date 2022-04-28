/*
  Copyright (C) 2021, University of Pennsylvania
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities
  \brief Convert PENNPet Explorer-type of sinogram to STIR-type
  The PENNPet projdata skip over the gaps, the --gaps options adds them back to
  STIR's projdata. Without the --gaps flag the output will match the input (as closely as possibly).
  \author Nikos Efthimiou

  \par Usage:
  \code
  --output-filename <filename>
  --data-to-fit <filename>
  --data-to-scale <filename>
  \endcode
*/

#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInterfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include <iostream>
#include <string>
#include "stir/recon_buildblock/BinNormalisationFromProjData.h"
#include "stir/TrivialDataSymmetriesForViewSegmentNumbers.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/LORCoordinates.h" 


#include "liblor.h"
#include "libimagio++.h"

static void
print_usage_and_exit(const char * const prog_name)
{
    std::cerr << "\nUsage:\n" << prog_name << "\\\n"
              << "\t--output <filename>\\\n"
              << "\t--template <filename>\\\n"
              << "\t--input <filename>\\\n"
              << "\t--add-gaps\\\n"
              << "\t--template_with_gaps\\\n"
              << "END:=\n";

    exit(EXIT_FAILURE);
}

USING_NAMESPACE_STIR

//! Regurn true if the ring is a gap (between 0 and 16)
bool is_gap(int r)
{
    int d = r % 56;
    if (d >= 0 && d < 16 )
        return true;
    return false;
}

int main(int argc, const char *argv[])
{
    const char * const prog_name = argv[0];

    std::string output_filename;
    std::string input_filename;
    std::string template_filename;
    std::string gtemplate_filename;
    bool addgaps = false;

    // option processing
    while (argc>1 && argv[1][1] == '-')
    {
        if (strcmp(argv[1], "--output")==0)
        {
            output_filename = (argv[2]);
            argc-=2; argv +=2;
        }
        else if (strcmp(argv[1], "--template")==0)
        {
            template_filename = argv[2];
            argc-=2; argv +=2;
        }
        else if (strcmp(argv[1], "--input") ==0)
        {
            input_filename = argv[2];
            argc-=2; argv +=2;
        }
        else if (strcmp(argv[1], "--template_with_gaps") ==0)
        {
            gtemplate_filename = argv[2];
            argc-=2; argv +=2;
        }
        else if (strcmp(argv[1], "--add-gaps") ==0)
        {
            addgaps = true;
            argc-=1; argv +=1;
        }
        else
        {
            std::cerr << "\nUnknown option: " << argv[1];
            print_usage_and_exit(prog_name);
        }
    }

    if (argc > 1)
    {
        std::cerr << "Command line should contain only options\n";
        print_usage_and_exit(prog_name);
    }

    const stir::shared_ptr<const ProjData> template_projdata_sptr =
            ProjData::read_from_file(template_filename);

    stir::shared_ptr<const ProjData> gtemplate_projdata_sptr;

    if(addgaps)
        gtemplate_projdata_sptr = ProjData::read_from_file(gtemplate_filename);

    shared_ptr<ProjData> output_projdata_sptr;

    if(addgaps)
        output_projdata_sptr.reset(new ProjDataInterfile (gtemplate_projdata_sptr->get_exam_info_sptr(),
                                                          gtemplate_projdata_sptr->get_proj_data_info_sptr(), output_filename));
    else
        output_projdata_sptr.reset(new ProjDataInterfile (template_projdata_sptr->get_exam_info_sptr(),
                                                          template_projdata_sptr->get_proj_data_info_sptr(), output_filename));


    const shared_ptr<const ProjDataInfoCylindricalNoArcCorr> out_projdata_info_sptr =
            dynamic_pointer_cast<ProjDataInfoCylindricalNoArcCorr>(output_projdata_sptr->get_proj_data_info_sptr()->create_shared_clone());


    InputImagioFile inputSinoFile;
    if(!inputSinoFile.open(input_filename))
    {
        std::cerr<<"Cannot read input file: "<< input_filename <<" Abort."<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    pet_mhdr mh_isino;
    mh_isino = inputSinoFile.mainHeader();
    llor::SinoMap sinoMap;
    if(!sinoMap.init(mh_isino))
    {
        std::cerr <<"Input list geometry is invalid. Abort."<<std::endl;
        std::exit(EXIT_FAILURE);
    }

    //    const int isliceBeg = 0;
    //    const int isliceEnd = mh_isino.nslice;
    //    const int minTilt = (mh_isino.ntilt ? 1 : 0);
    const int sliceSize = mh_isino.numang * mh_isino.numray;
    //    std::cout << mh_isino.numray << std::endl;
    //    const float angle_spacing = out_projdata_info_sptr->get_phi(Bin(0,1,0, 0, 0))-
    //            out_projdata_info_sptr->get_phi(Bin(0,0,0,0, 0));
    //    const float eff_ring_radius = out_projdata_info_sptr->get_scanner_sptr()->get_effective_ring_radius();

#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int segment_num = out_projdata_info_sptr->get_min_segment_num();
         segment_num<= out_projdata_info_sptr->get_max_segment_num(); ++segment_num)
    {
        int min_axial_pos = output_projdata_sptr->get_min_axial_pos_num(segment_num);
        int max_axial_pos = output_projdata_sptr->get_max_axial_pos_num(segment_num);

        SegmentBySinogram<float> seg(output_projdata_sptr->get_empty_segment_by_sinogram(segment_num));

        for(int i_axial = min_axial_pos; i_axial <= max_axial_pos; ++i_axial)
        {
            int _d1, _d2, _dr1, _dr2;
            int _cur_tilt, _cur_slice, _cur_phi, _cur_rad;
#if STIR_TOF
            Bin _tmp_bin(segment_num, 0, i_axial, 0, 0, 0.f);
#else
            Bin _tmp_bin(segment_num, 0, i_axial, 0, 0.f);
#endif
            out_projdata_info_sptr->get_det_pair_for_bin(_d1, _dr1, _d2, _dr2, _tmp_bin);
            //            const double ring_spacing_in = 1/out_projdata_info_sptr->get_ring_spacing();
            //            const double offset = 0.5*ring_spacing_in;

            //if not gap
            if(addgaps)
            {
                if(!is_gap(_dr1) && !is_gap(_dr2))
                {
                    _dr1 -= (static_cast<int>(_dr1/56) + 1) * 16; // a ring at the smaller geometry
                    _dr2 -= (static_cast<int>(_dr2/56) + 1) * 16; // a ring at the smaller geometry
//                    _d1 -= static_cast<int>(_d1/32);
//                    _d2 -= static_cast<int>(_d2/32); // a detector in the smaller geometry
                }
                else //Leave this pair 0 - is a gap
                    continue;

            }


            sinoMap.getSinoCoordsForCrystals(
                        0, _d1, _dr1,
                        0, _d2, _dr2,
                        &_cur_tilt, &_cur_slice, &_cur_phi, &_cur_rad, 0);

            std::vector<float> tmp_slice(sliceSize);
            if(!inputSinoFile.readSlice(1, _cur_tilt, _cur_slice, &tmp_slice[0]))
            {
                std::cerr <<"Cannot read input data. Abort."<<std::endl;
                std::exit(EXIT_FAILURE);
            }

            //norm_seg.emplace(_cur_slice, tmp_slice);
            Sinogram<float> cur_sino = seg.get_sinogram(i_axial);
            for(int i_phi = 0; i_phi <= output_projdata_sptr->get_max_view_num(); ++i_phi)
            {
                for(int i_tang = output_projdata_sptr->get_min_tangential_pos_num();
                    i_tang <= output_projdata_sptr->get_max_tangential_pos_num();
                    ++i_tang)
                {
                    float value = 0.f;
#ifdef STIR_TOF
                    Bin tmp_bin(segment_num, i_phi, i_axial, i_tang, 0, value);
#else
                    Bin tmp_bin(segment_num, i_phi, i_axial, i_tang, value);
#endif

                    if (i_tang < template_projdata_sptr->get_min_tangential_pos_num() ||
                            i_tang > template_projdata_sptr->get_max_tangential_pos_num())
                    {
                        cur_sino[i_phi][i_tang] = 0;
                        continue;
                    }
                    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_sino;
                    LORInCylinderCoordinates<float> lor_cyl;
                    int d1, d2, dr1, dr2;

                    out_projdata_info_sptr->get_det_pair_for_bin(d1, dr1, d2, dr2, tmp_bin);

                    if(addgaps)
                    {
                        if ((d1 % 33 == 0) ||
                                (d2 % 33 == 0)  )
                        {
                            cur_sino[i_phi][i_tang] = 0;
                            continue;
                        }

                        d1 -= static_cast<int>(d1/33) + 1;
                        d2 -= static_cast<int>(d2/33) + 1; // a detector in the smaller geometry
                    }

                    int cur_tilt, cur_slice, cur_phi, cur_rad;

                    sinoMap.getSinoCoordsForCrystals(
                                0, d1, _dr1,
                                0, d2, _dr2,
                                &cur_tilt, &cur_slice, &cur_phi, &cur_rad, 0);

                    cur_sino[i_phi][i_tang] = tmp_slice[cur_phi * mh_isino.numray + cur_rad];
                }
            }
            seg.set_sinogram(cur_sino, i_axial);
        }
        output_projdata_sptr->set_segment(seg);
    }


    return EXIT_SUCCESS;
}







