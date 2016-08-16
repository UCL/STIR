/*
  Copyright (C) 2001- 2009, Hammersmith Imanet Ltd
  This file is part of STIR.

  This file is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2.0 of the License, or
  (at your option) any later version.

  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  See STIR/LICENSE.txt for details
*/

/*!
  \brief Test class for exporting and importing ProjData
  as arrays.
  \ingroup test
  \author Nikos Efthimiou
*/

#include "stir/utilities.h"
#include "stir/RunTests.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Scanner.h"
#include "stir/ExamInfo.h"
#include "stir/SegmentByView.h"
#include "stir/Array.h"
#include "stir/IO/write_data.h"
#include "stir/IO/read_data.h"
#include "stir/info.h"

#include <boost/format.hpp>

#include "stir/ProjData.h"
#include "stir/DynamicProjData.h"

#include <fstream>
#include <iostream>
#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::string;
#endif

START_NAMESPACE_STIR

class ExportArrayTests : public RunTests
{
public:
    void run_tests();

protected:
    void test_static_data();
    void test_dynamic_data();
};

void ExportArrayTests :: run_tests()
{
    test_static_data();
    test_dynamic_data();
}


//!
//! \brief ExportArrayTests::test_dynamic_data
//! \details This test will try to write DynamicProjData in a 2D array.
//! Where, each 1D row will correspond to one frame. Then the data will be
//! written and retrieved from the disk, resorted in DynamicProjData and
//! compared with the original. The test will fails if the array retrieved
//! from the disk is of different size from the original.
void ExportArrayTests::test_dynamic_data()
{
    info("Initialising...");
    //. Create ProjData

    //- ProjDataInfo
    //-- Scanner
    shared_ptr<Scanner> test_scanner_sptr( new Scanner(Scanner::Siemens_mMR));

    //-- ExamInfo
    shared_ptr<ExamInfo> test_exam_info_sptr(new ExamInfo());
    // TODO, Currently all stir::Scanner types are PET.
    test_exam_info_sptr->imaging_modality = ImagingModality::PT;

    info("...new DynamicProjData...");
    shared_ptr<DynamicProjData> test_dynamic_projData_first_sptr (new DynamicProjData(test_exam_info_sptr));

    info ("...ProjDataInfo...");
    shared_ptr<ProjDataInfo> tmp_proj_data_info_sptr(
                ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                              1,
                                              1, /* Reduce the number of segments */
                                              test_scanner_sptr->get_max_num_views(),
                                              test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                              false));

    info("Resizing the DynamicProjData... ");
    int num_of_gates = 3;
    test_dynamic_projData_first_sptr->resize( num_of_gates);


    for (int i_gate = 0; i_gate < num_of_gates; i_gate++)
    {
        info(boost::format("Allocating and filling the %1% gate... ") % i_gate);

        shared_ptr<ProjData> test_proj_data_gate_ptr(
                    new ProjDataInMemory(test_exam_info_sptr,
                                         tmp_proj_data_info_sptr));

        for (int segment_num = test_proj_data_gate_ptr->get_min_segment_num();
             segment_num <= test_proj_data_gate_ptr->get_max_segment_num();
             ++segment_num)
        {
            SegmentByView<float> segment_by_view_data =
                    test_proj_data_gate_ptr->get_segment_by_view(segment_num);

            segment_by_view_data.fill(segment_num + (i_gate * 1000));

            if (!(test_proj_data_gate_ptr->set_segment(segment_by_view_data) == Succeeded::yes))
                warning("Error set_segment %d\n", segment_num);
        }
        info("Populating the Dynamic ProjData... ");
        test_dynamic_projData_first_sptr->set_proj_data_sptr(test_proj_data_gate_ptr, (i_gate+1));
    }

    // Get total size
    info("Getting sizes...");
    unsigned long int total_size = test_dynamic_projData_first_sptr->size_all();

    unsigned long int total_gates = test_dynamic_projData_first_sptr->get_num_gates();

    unsigned long int projdata_size = test_dynamic_projData_first_sptr->get_projData_size();
    info(boost::format("Total size: %1%, number of gates: %2%, size of projdata %3%") % total_size % total_gates %
         projdata_size);
    // Allocate 2D array to store the data.
    info("Allocating first (output) 2D array...");
    Array<2, float> test_array_first (IndexRange2D(0, total_gates, 0, projdata_size));
    test_array_first.fill(-100);
    Array<2, float>::full_iterator test_out_array_first_iter = test_array_first.begin_all();

    info("Allocating second (input) 2D array...");
    Array<2, float> test_array_second (IndexRange2D(0, total_gates, 0, projdata_size));
    test_array_second.fill(-100);
    Array<2, float>::full_iterator test_array_second_iter = test_array_second.begin_all();

    // Copy data to array.
    info("Copying dynamic projdata to first array ...");
    test_dynamic_projData_first_sptr->copy_to< Array<2,float>::full_iterator >( test_out_array_first_iter);

    // Write the array to disk
    info("Writing data to disk ... ");
    {
        std::ofstream outfile;
        outfile.open("./temp.data");
        write_data<2,  std::ofstream , float> ( outfile, test_array_first);
        outfile.close();
    }

    // Load it back
    info("Loading back the array...");
    {
        std::ifstream infile;
        infile.open("./temp.data");
        read_data<2, std::ifstream, float>(infile, test_array_second);
        infile.close();
    }

    // compare it with the previous array
    check_if_equal(test_array_first, test_array_second, "Array loaded from disk is different from original.");

    // Convert it to ProjData
    info("Copying data from array to dynamic projdata ...");

    shared_ptr<DynamicProjData> test_dynamic_projData_second_sptr (new DynamicProjData(test_exam_info_sptr,
                                                                                       num_of_gates));

    for (int i_gate = 0; i_gate < num_of_gates; i_gate++)
    {
        info(boost::format("Allocating and filling the %1% gate... ") % i_gate);

        shared_ptr<ProjData> test_proj_data_gate_ptr(
                    new ProjDataInMemory(test_exam_info_sptr,
                                         tmp_proj_data_info_sptr));

        info("Populating the Dynamic ProjData... ");
        test_dynamic_projData_second_sptr->set_proj_data_sptr(test_proj_data_gate_ptr, (i_gate+1));
    }

    test_dynamic_projData_second_sptr->fill_from<Array<2,float>::full_iterator>(test_array_second_iter);

    info ("Checking if data are the same...");
    for(int i_gate = 0; i_gate < num_of_gates; i_gate++)
    {
        shared_ptr<ProjData> _first_sptr (test_dynamic_projData_first_sptr->get_proj_data_sptr((i_gate+1)));
        shared_ptr<ProjData> _second_sptr(test_dynamic_projData_second_sptr->get_proj_data_sptr((i_gate+1)));

        for (int segment_num = _first_sptr->get_min_segment_num();
             segment_num <= _first_sptr->get_max_segment_num();
             ++segment_num)
        {
            SegmentByView<float> _first_segment_by_view_data =
                    _first_sptr->get_segment_by_view(segment_num);

            SegmentByView<float> _second_segment_by_view_data =
                    _second_sptr->get_segment_by_view(segment_num);

            for (int view_num = _first_segment_by_view_data.get_min_view_num();
                 view_num <= _first_segment_by_view_data.get_max_view_num();
                 ++view_num)
            {
                Viewgram<float> _first_view = _first_segment_by_view_data.get_viewgram(view_num);
                Viewgram<float> _second_view = _second_segment_by_view_data.get_viewgram(view_num);

                for (int axial = _first_view.get_min_axial_pos_num();
                     axial <= _first_view.get_max_axial_pos_num();
                     ++axial)
                {
                    for (int s = _first_view.get_min_tangential_pos_num();
                         s <= _first_view.get_max_tangential_pos_num();
                         ++s)
                    {
                        check_if_equal(_first_view[axial][s], _second_view[axial][s], "Different data loaded");
                    }
                }
            }
        }
    }

}

//!
//! \brief ExportArrayTests::test_static_data
//! \details This test will try to fill a 1D array with data from
//! a ProjData object. Then the array will be written and retrieved from
//! the disk, the data will be sorted back to a ProjData object and be
//! compared to the original. The test will fails if the array retrieved
//! from the disk is of different size from the original.
void ExportArrayTests :: test_static_data()
{
    //. Create ProjData

    //- ProjDataInfo
    //-- Scanner
    shared_ptr<Scanner> test_scanner_sptr( new Scanner(Scanner::Siemens_mMR));

    //-- ExamInfo
    shared_ptr<ExamInfo> test_exam_info_sptr(new ExamInfo());
    // TODO, Currently all stir::Scanner types are PET.
    test_exam_info_sptr->imaging_modality = ImagingModality::PT;

    //-
    shared_ptr<ProjDataInfo> tmp_proj_data_info_sptr(
                ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                              1,
                                              1,
                                              test_scanner_sptr->get_max_num_views(),
                                              test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                              false));

    shared_ptr<ProjData> test_proj_data_ptr(
                new ProjDataInMemory(test_exam_info_sptr,
                                     tmp_proj_data_info_sptr));


    shared_ptr<ProjData> test_proj_data_second_ptr(
                new ProjDataInMemory(test_exam_info_sptr,
                                     tmp_proj_data_info_sptr));

    //. Fill ProjData with the number of the segment

    for (int segment_num = test_proj_data_ptr->get_min_segment_num();
         segment_num <= test_proj_data_ptr->get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> segment_by_view_data =
                test_proj_data_ptr->get_segment_by_view(segment_num);

        segment_by_view_data.fill(segment_num);

        if (!(test_proj_data_ptr->set_segment(segment_by_view_data) == Succeeded::yes))
            warning("Error set_segment %d\n", segment_num);
    }


    //. Pass it to an array

    //- Get the total size of the ProjData

    unsigned long int total_size = test_proj_data_ptr->size_all();

    //- Allocate 1D array and get iterator

    info("Allocating arrays ...");
    Array<1,float> test_out_array(0, total_size);
    Array<1,float>::full_iterator test_out_array_iter = test_out_array.begin_all();

    Array<1,float> test_in_array(0, total_size);
    Array<1,float>::full_iterator test_in_array_iter = test_in_array.begin_all();

    //-
    info("Copying data to array ...");
    test_proj_data_ptr->copy_to< Array<1,float>::full_iterator >( test_out_array_iter);

    // Write the array to disk
    info("Writing data to disk ... ");
    {
        std::ofstream outfile;
        outfile.open("./temp.data");
        write_data<1,  std::ofstream , float> ( outfile, test_out_array);
        outfile.close();
    }

    // Load it back
    info("Loading back the array...");
    {
        std::ifstream infile;
        infile.open("./temp.data");
        read_data<1, std::ifstream, float>(infile, test_in_array);
        infile.close();
    }

    // compare it with the previous array
    check_if_equal(test_in_array, test_out_array, "Array loaded from disk is different from original.");

    // Convert it to ProjData
    info("Copying data ...");
    test_proj_data_second_ptr->fill_from<Array<1,float>::full_iterator>(test_in_array_iter);

    info ("Checking if data are the same...");
    for (int segment_num = test_proj_data_ptr->get_min_segment_num();
         segment_num <= test_proj_data_ptr->get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> segment_by_view_data =
                test_proj_data_ptr->get_segment_by_view(segment_num);

        for (int view_num=segment_by_view_data.get_min_view_num();
             view_num<=segment_by_view_data.get_max_view_num();
             view_num++)
        {
            Viewgram<float> view = segment_by_view_data.get_viewgram(view_num);

            for (int axial = view.get_min_axial_pos_num();
                 axial < view.get_max_axial_pos_num();
                 axial++)
            {
                for (int s = view.get_min_tangential_pos_num();
                     s < view.get_max_tangential_pos_num();
                     s++)
                {
                    check_if_equal(view[axial][s], (float)segment_num, "Different data loaded");
                }
            }
        }
    }
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR
int main(int argc, char **argv)
{
    ExportArrayTests tests;
    tests.run_tests();
    return tests.main_return_value();
}
