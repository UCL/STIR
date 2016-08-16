/*
    Copyright (C) 2016 University College London

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
#include "stir/info.h"

#include <boost/format.hpp>

#include "stir/ProjData.h"
#include "stir/DynamicProjData.h"


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

    info("Creating test DynamicProjData...");
    shared_ptr<DynamicProjData> test_dynamic_projData_sptr (new DynamicProjData(test_exam_info_sptr));

    shared_ptr<ProjDataInfo> tmp_proj_data_info_sptr(
                ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                              1,
                                              1, /* Reduce the number of segments */
                                              test_scanner_sptr->get_max_num_views(),
                                              test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                              false));

    const int num_of_gates = 3;
    info(boost::format("Resizing the DynamicProjData for %1% gates... ") % num_of_gates);
    test_dynamic_projData_sptr->resize( num_of_gates);

    for (int i_gate = 1; i_gate <= num_of_gates; i_gate++)
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

            // 1000 is an arbitary number to distiguish data in different gates.
            segment_by_view_data.fill(segment_num + (i_gate * 1000));

            if (!(test_proj_data_gate_ptr->set_segment(segment_by_view_data) == Succeeded::yes))
                warning("Error set_segment %d\n", segment_num);
        }

        info("Populating the Dynamic ProjData... ");
        test_dynamic_projData_sptr->set_proj_data_sptr(test_proj_data_gate_ptr, i_gate);
    }

    const std::size_t total_size = test_dynamic_projData_sptr->size_all();
    const int total_gates = static_cast<int>(test_dynamic_projData_sptr->get_num_proj_data());
    const int projdata_size = static_cast<int>(test_dynamic_projData_sptr->get_proj_data_size());

    info(boost::format("Total size: %1%, number of gates: %2%, size of projdata %3%") % total_size % total_gates %
         projdata_size);
    // Allocate 2D array to store the data.
    info("Allocating test array...");
    Array<2, float> test_array (IndexRange2D(0, total_gates, 0, projdata_size));
    test_array.fill(-1);
    Array<2, float>::full_iterator test_array_iter = test_array.begin_all();

    // Copy data to array.
    info("Copying test dynamic projdata to array ...");
    test_dynamic_projData_sptr->copy_to< Array<2,float>::full_iterator >( test_array_iter);

    // Convert it to ProjData
    info("Copying data from array to check dynamic projdata ...");

    shared_ptr<DynamicProjData> check_dynamic_projData_sptr (new DynamicProjData(test_exam_info_sptr,
                                                                                       num_of_gates));

    for (int i_gate = 0; i_gate < num_of_gates; i_gate++)
    {
        info(boost::format("Allocating and filling the %1% gate... ") % (i_gate+1));

        shared_ptr<ProjData> test_proj_data_gate_ptr(
                    new ProjDataInMemory(test_exam_info_sptr,
                                         tmp_proj_data_info_sptr));

        info("Populating the Dynamic ProjData... ");
        check_dynamic_projData_sptr->set_proj_data_sptr(test_proj_data_gate_ptr, (i_gate+1));
    }

    check_dynamic_projData_sptr->fill_from<Array<2,float>::full_iterator>(test_array_iter);

    info ("Checking if data are the same...");
    for(int i_gate = 1; i_gate <= num_of_gates; i_gate++)
    {
        shared_ptr<ProjData> _test_projdata_sptr (test_dynamic_projData_sptr->get_proj_data_sptr(i_gate));
        shared_ptr<ProjData> _check_projdata_sptr(check_dynamic_projData_sptr->get_proj_data_sptr(i_gate));

        for (int segment_num = _test_projdata_sptr->get_min_segment_num();
             segment_num <= _test_projdata_sptr->get_max_segment_num();
             ++segment_num)
        {
            SegmentByView<float> _test_segment_by_view_data =
                    _test_projdata_sptr->get_segment_by_view(segment_num);

            SegmentByView<float> _check_segment_by_view_data =
                    _check_projdata_sptr->get_segment_by_view(segment_num);

            for (int view_num = _test_segment_by_view_data.get_min_view_num();
                 view_num <= _test_segment_by_view_data.get_max_view_num();
                 ++view_num)
            {
                Viewgram<float> _test_view = _test_segment_by_view_data.get_viewgram(view_num);
                Viewgram<float> _check_view = _check_segment_by_view_data.get_viewgram(view_num);

                for (int axial = _test_view.get_min_axial_pos_num();
                     axial <= _test_view.get_max_axial_pos_num();
                     ++axial)
                {
                    for (int s = _test_view.get_min_tangential_pos_num();
                         s <= _test_view.get_max_tangential_pos_num();
                         ++s)
                    {
                        check_if_equal(_test_view[axial][s], _check_view[axial][s], "Test ProjData different from check ProjData.");
                        check_if_equal(_check_view[axial][s], (float) (segment_num + (i_gate * 1000)), "Check ProjData different from segment number.");
                    }
                }
            }
        }
    }

}

//!
//! \brief ExportArrayTests::test_static_data
//! \details This test will chech if projection data copied to arrays are the
//! same when copied back to projdata.
//!
void ExportArrayTests :: test_static_data()
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

    //-
    shared_ptr<ProjDataInfo> tmp_proj_data_info_sptr(
                ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                              1,
                                              1,
                                              test_scanner_sptr->get_max_num_views(),
                                              test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                              false));

    shared_ptr<ProjData> test_proj_data_sptr(
                new ProjDataInMemory(test_exam_info_sptr,
                                     tmp_proj_data_info_sptr));


    shared_ptr<ProjData> check_proj_data_sptr(
                new ProjDataInMemory(test_exam_info_sptr,
                                     tmp_proj_data_info_sptr));

    //. Fill ProjData with the number of the segment
    info("Filling test ProjData with the segment number ... ");

    for (int segment_num = test_proj_data_sptr->get_min_segment_num();
         segment_num <= test_proj_data_sptr->get_max_segment_num();
         ++segment_num)
    {
        info(boost::format("Segment: %1% ") % segment_num);
        SegmentByView<float> segment_by_view_data =
                test_proj_data_sptr->get_segment_by_view(segment_num);

        segment_by_view_data.fill(segment_num);

        if (!(test_proj_data_sptr->set_segment(segment_by_view_data) == Succeeded::yes))
            warning("Error set_segment %d\n", segment_num);
    }

    //- Get the total size of the ProjData

    int total_size = static_cast<int>(test_proj_data_sptr->size_all());

    //- Allocate 1D array and get iterator

    info("Allocating array ...");
    Array<1,float> test_array(0, total_size);
    Array<1,float>::full_iterator test_array_iter = test_array.begin_all();

    //-
    info("Copying from ProjData to array ...");
    test_proj_data_sptr->copy_to< Array<1,float>::full_iterator >( test_array_iter);

    // Convert it back to ProjData
    info("Copying from array to a new ProjData ...");
    check_proj_data_sptr->fill_from<Array<1,float>::full_iterator>(test_array_iter);

    info ("Checking if new and old ProjData are the same...");
    for (int segment_num = test_proj_data_sptr->get_min_segment_num();
         segment_num <= test_proj_data_sptr->get_max_segment_num();
         ++segment_num)
    {
        SegmentByView<float> test_segment_by_view_data =
                test_proj_data_sptr->get_segment_by_view(segment_num);

        SegmentByView<float> check_segment_by_view_data =
                check_proj_data_sptr->get_segment_by_view(segment_num);

        for (int view_num = test_segment_by_view_data.get_min_view_num();
             view_num<=test_segment_by_view_data.get_max_view_num();
             ++view_num)
        {
            Viewgram<float> test_view = test_segment_by_view_data.get_viewgram(view_num);
            Viewgram<float> check_view = check_segment_by_view_data.get_viewgram(view_num);

            for (int axial = test_view.get_min_axial_pos_num();
                 axial <= test_view.get_max_axial_pos_num();
                 ++axial)
            {
                for (int s = test_view.get_min_tangential_pos_num();
                     s <= test_view.get_max_tangential_pos_num();
                     ++s)
                {
                    check_if_equal(test_view[axial][s], check_view[axial][s], "Test ProjData different from check ProjData.");
                    check_if_equal(check_view[axial][s], (float)segment_num, "Check ProjData different from segment number.");
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
