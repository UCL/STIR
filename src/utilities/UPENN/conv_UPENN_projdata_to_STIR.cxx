/*
  Copyright (C) 2021, University of Pennsylvania
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities
  \brief Convert PennPET Explorer-type of sinogram to STIR-type
  The PennPET projdata skip over the gaps, the --gaps options adds them back to
  STIR's projdata. Without the --gaps flag the output will match the input (as closely as possibly).
  --inv: should be used for the attenuation correction
  --up_treshold should be used with normalisation (typically 13)
  \author Nikos Efthimiou

  \par Usage:
  \code
    --output <filename>
    --template <filename>
    --input <filename>
    --add-gaps
    --template_with_gaps
    --inv
    --up_thresshold <value>
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
              << "\t--inv\\\n"
              << "\t--up_thresshold <value>\\\n"
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
    bool invert = false;
    bool max = false;
    bool up_thresshold = false;
    float up_thresshold_value = 13.f;

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
        else if (strcmp(argv[1], "--inv") ==0)
        {
            invert = true;
            argc-=1; argv +=1;
        }
        else if (strcmp(argv[1], "--max") ==0)
        {
            max = true;
            argc-=1; argv +=1;
        }
        else if (strcmp(argv[1], "--up_thresshold") ==0)
        {
            up_thresshold = true;
            up_thresshold_value = atof(argv[2]);
            argc-=2; argv +=2;
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

    long int fillvalue = 0;
    if (max)
        fillvalue = 1000000000;
#ifdef STIR_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int iseg = out_projdata_info_sptr->get_min_segment_num();
         iseg <= out_projdata_info_sptr->get_max_segment_num(); ++iseg)
    {
        SegmentByView<float> d = out_projdata_info_sptr->get_empty_segment_by_view(iseg);
        d.fill(fillvalue);
        output_projdata_sptr->set_segment(d);
    }

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

    const int isliceEnd = mh_isino.nslice;
    const int sliceSize = mh_isino.numang * mh_isino.numray;
    std::cout << mh_isino.numang << " " << mh_isino.numray << std::endl;
    std::cout << isliceEnd << std::endl;


    //#ifdef STIR_OPENMP
    //#pragma omp parallel for schedule(dynamic)
    //#endif
#ifdef STIR_OPENMP
#ifdef _WIN32
#pragma omp parallel for
#else
#pragma omp parallel for schedule(dynamic)
#endif
#endif
    for (int iSlice = 0;
         iSlice < sinoMap.nslice(); ++iSlice)
    {

        int  _dr1, _dr2;
        int dr1, dr2;// with gaps

        sinoMap.getCrystalZsForSinoSlice(iSlice, &_dr1, &_dr2);
        std::vector<float> tmp_slice(sliceSize);
#ifdef STIR_OPENMP
#pragma omp critical(LISTMODEIO)
#endif
        {
            if(!inputSinoFile.readSlice(1, 1, iSlice, &tmp_slice[0]));
            {
                //                std::cerr <<"Cannot read input data. Abort."<<std::endl;
                //                std::exit(EXIT_FAILURE);
                //                more = false;
            }
        }

        if(addgaps)
        {
            int r = _dr1 * 0.025f;
            dr1 = _dr1 + 16*(r + 1);
            r = _dr2 * 0.025f;
            dr2 = _dr2 + 16*(r + 1); // a ring at the smaller geometry
        }
        else //Leave this pair 0 - is a gap
        {
            dr1 = _dr1;
            dr2 = _dr2;
        }

        int cur_seg, cur_axial;
        out_projdata_info_sptr->get_segment_axial_pos_num_for_ring_pair(cur_seg, cur_axial,
                                                                        dr1, dr2);
        std::cout << cur_seg << std::endl;

        Sinogram<float> cur_sino(output_projdata_sptr->get_empty_sinogram(cur_axial, cur_seg));
        cur_sino.fill(fillvalue);

        for(int i_phi = 0; i_phi <= template_projdata_sptr->get_max_view_num(); ++i_phi)
        {
            for(int i_tang = 0;
                i_tang < template_projdata_sptr->get_num_tangential_poss();
                ++i_tang)
            {
                int d1, d2;
                int _d1, _d2;

                sinoMap.getCrystalXsForSinoPhiRadMash(i_phi, i_tang, 0, &_d1, &_d2);

                if(addgaps)
                {
                    d1 = _d1 + static_cast<int>(_d1 * 0.03125f) + 1; // add gap
                    d2 = _d2 + static_cast<int>(_d2 * 0.03125f) + 1; // add gap
                }
                else
                {
                    d1 = _d1;
                    d2 = _d2;
                }

                Bin tmp_bin;

                out_projdata_info_sptr->get_bin_for_det_pair(tmp_bin, d1, dr1, d2, dr2);

                const int index = i_phi * mh_isino.numray + i_tang;
                float val = tmp_slice[index];

                if(up_thresshold)
                {
                    if (val > up_thresshold_value)
                        val = up_thresshold_value;
                }


                if(invert) //for attenuation correction
                {
                    if (val != 0)
                    {
                        cur_sino[tmp_bin.view_num()][tmp_bin.tangential_pos_num()] = 1.f/val;
                    }
                    else
                        cur_sino[tmp_bin.view_num()][tmp_bin.tangential_pos_num()] = 0.f;
                }
                else
                {
                    if (val != 0)
                    {
                        cur_sino[tmp_bin.view_num()][tmp_bin.tangential_pos_num()] = val;
                    }
                    else
                        cur_sino[tmp_bin.view_num()][tmp_bin.tangential_pos_num()] = fillvalue;
                }

            }
        }
        output_projdata_sptr->set_sinogram(cur_sino);
    }
    return EXIT_SUCCESS;
}
