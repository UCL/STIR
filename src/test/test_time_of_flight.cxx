/*
    Copyright (C) 2016, UCL
    Copyright (C) 2016, University of Hull
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
  \ingroup test
  \brief Test class for Time-Of-Flight
  \author Nikos Efthimiou
*/
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingSeparateProjectors.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/RelatedViewgrams.h"
//#include "stir/geometry/line_distances.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "boost/lexical_cast.hpp"

#include "stir/info.h"
#include "stir/warning.h"

START_NAMESPACE_STIR

//! A helper class to keep the combination of a view, a segment and
//! a key tight.
//! \author Nikos Efthimiou
class cache_index{
public:
    cache_index() {
        view_num = 0;
        seg_num = 0;
        key = 0;
    }

    inline bool operator==(const cache_index& Y) const
    {
        return view_num == Y.view_num &&
                seg_num == Y.seg_num &&
                key == Y.key;
    }

    inline bool operator!=(const cache_index& Y) const
    {
        return !(*this == Y);
    }

    inline bool operator< (const cache_index& Y) const
    {
        return view_num < Y.view_num &&
                seg_num < Y.seg_num &&
                key < Y.key;
    }

    int view_num;
    int seg_num;
    boost::uint32_t key;
};

// Helper class.
class FloatFloat{
public:
    FloatFloat() { float1 = 0.f; float2 = 0.f;}
    float float1;
    float float2;
};

class TOF_Tests : public RunTests
{
public:
    void run_tests();

private:

    void test_tof_proj_data_info();

    void test_tof_geometry_1();

    void test_tof_geometry_2();

    //! This checks peaks a specific bin, finds the LOR and applies all the
    //! kernels of all available timing positions.
    void test_tof_kernel_application();

    void
    export_lor(ProjMatrixElemsForOneBin& probabilities,
               const CartesianCoordinate3D<float>& point1,
               const CartesianCoordinate3D<float>& point2,int current_id);

    void
    export_lor(ProjMatrixElemsForOneBin& probabilities,
               const CartesianCoordinate3D<float>& point1,
               const CartesianCoordinate3D<float>& point2,int current_id,
               ProjMatrixElemsForOneBin& template_probabilities);

    shared_ptr<Scanner> test_scanner_sptr;
    shared_ptr<ProjDataInfo> test_proj_data_info_sptr;
    shared_ptr<DiscretisedDensity<3, float> > test_discretised_density_sptr;
    shared_ptr<ProjMatrixByBin> test_proj_matrix_sptr;
    shared_ptr<ProjectorByBinPair> projector_pair_sptr;
    shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_used_sptr;
};

void
TOF_Tests::run_tests()
{
    // New Scanner
    test_scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa));

    // New Proj_Data_Info
    const int test_tof_mashing_factor = 39; // to have 9 TOF bins (381/39=9)
    test_proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                                                 1,test_scanner_sptr->get_num_rings() -1,
                                                                 test_scanner_sptr->get_num_detectors_per_ring()/2,
                                                                 test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                                 /* arc_correction*/false));
    test_proj_data_info_sptr->set_tof_mash_factor(test_tof_mashing_factor);

    test_tof_proj_data_info();
//    test_tof_geometry_1();

    // New Discretised Density
    test_discretised_density_sptr.reset( new VoxelsOnCartesianGrid<float> (*test_proj_data_info_sptr, 1.f,
                                                                           CartesianCoordinate3D<float>(0.f, 0.f, 0.f),
                                                                           CartesianCoordinate3D<int>(-1, -1, -1)));
    // New ProjMatrix
    test_proj_matrix_sptr.reset(new ProjMatrixByBinUsingRayTracing());
    dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_proj_matrix_sptr.get())->set_num_tangential_LORs(1);
    dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_proj_matrix_sptr.get())->set_up(test_proj_data_info_sptr, test_discretised_density_sptr);
//    test_proj_matrix_sptr->enable_tof(test_proj_data_info_sptr);

    shared_ptr<ForwardProjectorByBin> forward_projector_ptr(
                new ForwardProjectorByBinUsingProjMatrixByBin(test_proj_matrix_sptr));
    shared_ptr<BackProjectorByBin> back_projector_ptr(
                new BackProjectorByBinUsingProjMatrixByBin(test_proj_matrix_sptr));

    projector_pair_sptr.reset(
                new ProjectorByBinPairUsingSeparateProjectors(forward_projector_ptr, back_projector_ptr));
    projector_pair_sptr->set_up(test_proj_data_info_sptr, test_discretised_density_sptr);

    symmetries_used_sptr.reset(projector_pair_sptr->get_symmetries_used()->clone());

    // Deactivated it now because it takes a long time to finish.
    //        test_cache();

    test_tof_kernel_application();
}

void
TOF_Tests::test_tof_proj_data_info()
{
    const int correct_tof_mashing_factor = 39;
    const int num_timing_positions = 9;
    float correct_width_of_tof_bin = test_scanner_sptr->get_size_of_timing_pos() *
            test_proj_data_info_sptr->get_tof_mash_factor() * 0.299792458f/2;
    float correct_timing_locations[num_timing_positions] = {-360.201f/2, -280.156f/2, -200.111f/2, -120.067f/2, -40.022f/2, 40.022f/2,
                                          120.067f/2, 200.111f/2, 280.156f/2};

    check_if_equal(correct_tof_mashing_factor,
                   test_proj_data_info_sptr->get_tof_mash_factor(), "Different TOF mashing factor.");

    check_if_equal(num_timing_positions,
                   test_proj_data_info_sptr->get_num_tof_poss(), "Different number of timing positions.");

    for (int timing_num = test_proj_data_info_sptr->get_min_tof_pos_num(), counter = 0;
         timing_num <= test_proj_data_info_sptr->get_max_tof_pos_num(); ++ timing_num, counter++)
    {
        Bin bin(0, 0, 0, 0, timing_num, 1.f);

        check_if_equal(static_cast<double>(correct_width_of_tof_bin),
                       static_cast<double>(test_proj_data_info_sptr->get_sampling_in_k(bin)), "Error in get_sampling_in_k()");
        check_if_equal(static_cast<double>(correct_timing_locations[counter]),
                       static_cast<double>(test_proj_data_info_sptr->get_k(bin)), "Error in get_sampling_in_k()");
    }

    float total_width = test_proj_data_info_sptr->get_k(Bin(0,0,0,0,test_proj_data_info_sptr->get_max_tof_pos_num(),1.f))
            - test_proj_data_info_sptr->get_k(Bin(0,0,0,0,test_proj_data_info_sptr->get_min_tof_pos_num(),1.f))
            + test_proj_data_info_sptr->get_sampling_in_k(Bin(0,0,0,0,0,1.f));

    set_tolerance(static_cast<double>(0.005));
    check_if_equal(static_cast<double>(total_width), static_cast<double>(test_proj_data_info_sptr->get_coincidence_window_width()),
                   "Coincidence widths don't match.");


}

void
TOF_Tests::test_tof_geometry_1()
{

    float correct_scanner_length = test_scanner_sptr->get_ring_spacing() *
            test_scanner_sptr->get_num_rings() - test_proj_data_info_sptr->get_sampling_in_m(Bin(0,0,0,0,0));

    CartesianCoordinate3D<float> ez1_coord0, ez2_coord0, ez3_coord0, ez4_coord0, ez5_coord0;
    //ProjDataInfoCylindrical* proj_data_ptr =
    //        dynamic_cast<ProjDataInfoCylindrical*> (test_proj_data_info_sptr.get());

    int mid_seg = test_proj_data_info_sptr->get_num_segments()/2;

    int mid_axial_0 = (test_proj_data_info_sptr->get_min_axial_pos_num(0) +
                     test_proj_data_info_sptr->get_max_axial_pos_num(0)) /2;

    int mid_axial_mid_seg = (test_proj_data_info_sptr->get_min_axial_pos_num(mid_seg) +
                     test_proj_data_info_sptr->get_max_axial_pos_num(mid_seg)) /2;

    // Some easy to validate bins:
    Bin ez1_bin(0,0,0,0,0,1.f);
    Bin ez2_bin(0,0,mid_axial_0,0,0,1.f);
    Bin ez3_bin(mid_seg,0,mid_axial_mid_seg,0,0,1.f);
    Bin ez4_bin(0,0,test_proj_data_info_sptr->get_min_axial_pos_num(0),0,0,1.f);
    Bin ez5_bin(0,0,test_proj_data_info_sptr->get_max_axial_pos_num(0),0,0,1.f);

    // Get middle points
//    proj_data_ptr->get_LOR_middle_point(ez1_coord0, ez1_bin);
//    proj_data_ptr->get_LOR_middle_point(ez2_coord0, ez2_bin);
//    proj_data_ptr->get_LOR_middle_point(ez3_coord0, ez3_bin);
//    proj_data_ptr->get_LOR_middle_point(ez4_coord0, ez4_bin);
//    proj_data_ptr->get_LOR_middle_point(ez5_coord0, ez5_bin);

    // axial ez1 && ez4 should be -scanner_length/2.f
    check_if_equal(static_cast<double>(ez1_coord0.z()),
                   static_cast<double>(-correct_scanner_length/2.f),
                   "Min axial positions of mid-points don't look "
                   "reasonable.");

    check_if_equal(static_cast<double>(ez4_coord0.z()),
                   static_cast<double>(-correct_scanner_length/2.f),
                   "Min axial positions of mid-points don't look "
                   "reasonable.");

    // axial ez2 should be -ring_spacing/2
    check_if_equal(static_cast<double>(ez2_coord0.z()),
                   static_cast<double>(-test_scanner_sptr->get_ring_spacing()/2.f),
                   "[1]Central axial positions of mid-points don't look "
                   "reasonable.");

    // axial ez3 should be 0
    check_if_equal(static_cast<double>(ez3_coord0.z()),
                   static_cast<double>(-test_proj_data_info_sptr->get_m(Bin(mid_seg,0,0,0,0))),
                   "[2]Central axial positions of mid-points don't look "
                   "reasonable.");

    // axial ez5 should be scanner_length/2.f
    check_if_equal(static_cast<double>(ez5_coord0.z()),
                   static_cast<double>(correct_scanner_length/2.f),
                   "Max axial positions of mid-points don't look "
                   "reasonable.");

    //TODO: more tests for X and Y
}

void
TOF_Tests::test_tof_geometry_2()
{
    //float correct_scanner_length = test_scanner_sptr->get_ring_spacing() *
    //        test_scanner_sptr->get_num_rings() - test_proj_data_info_sptr->get_sampling_in_m(Bin(0,0,0,0,0));

    CartesianCoordinate3D<float> ez1_coord1, ez2_coord1, ez3_coord1, ez4_coord1, ez5_coord1;
    CartesianCoordinate3D<float> ez1_coord2, ez2_coord2, ez3_coord2, ez4_coord2, ez5_coord2;
    //ProjDataInfoCylindrical* proj_data_ptr =
    //        dynamic_cast<ProjDataInfoCylindrical*> (test_proj_data_info_sptr.get());

    int mid_seg = test_proj_data_info_sptr->get_num_segments()/2;

    int mid_axial_0 = (test_proj_data_info_sptr->get_min_axial_pos_num(0) +
                     test_proj_data_info_sptr->get_max_axial_pos_num(0)) /2;

    int mid_axial_mid_seg = (test_proj_data_info_sptr->get_min_axial_pos_num(mid_seg) +
                     test_proj_data_info_sptr->get_max_axial_pos_num(mid_seg)) /2;

    // Some easy to validate bins:
    Bin ez1_bin(0,0,0,0,0,1.f);
    Bin ez2_bin(0,0,mid_axial_0,0,0,1.f);
    Bin ez3_bin(mid_seg,0,mid_axial_mid_seg,0,0,1.f);
    Bin ez4_bin(0,0,test_proj_data_info_sptr->get_min_axial_pos_num(0),0,0,1.f);
    Bin ez5_bin(0,0,test_proj_data_info_sptr->get_max_axial_pos_num(0),0,0,1.f);

    // Get middle points
//    proj_data_ptr->get_LOR_as_two_points(ez1_coord1,ez1_coord2, ez1_bin);
//    proj_data_ptr->get_LOR_as_two_points(ez2_coord1,ez2_coord2, ez2_bin);
//    proj_data_ptr->get_LOR_as_two_points(ez3_coord1,ez3_coord2, ez3_bin);
//    proj_data_ptr->get_LOR_as_two_points(ez4_coord1,ez4_coord2, ez4_bin);
//    proj_data_ptr->get_LOR_as_two_points(ez5_coord1,ez5_coord2, ez5_bin);

    // TESTS TO COME.

    // TEST IF THE FLIPING IS OK.
}

void
TOF_Tests::test_tof_kernel_application()
{
    int seg_num = 0;
    int view_num = 0;
    int axial_num = 0;
    int tang_num = 0;

    ProjMatrixElemsForOneBin proj_matrix_row;
    HighResWallClockTimer t;
    std::vector<double> times_of_tofing;

    ProjDataInfoCylindrical* proj_data_ptr =
            dynamic_cast<ProjDataInfoCylindrical*> (test_proj_data_info_sptr.get());

    Bin this_bin(seg_num, view_num, axial_num, tang_num, 1.f);

    t.reset(); t.start();
    test_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, this_bin);
    t.stop();
    std::cerr<<"Execution time for nonTOF: "<<t.value() << std::endl;
//    export_lor(proj_matrix_row,
//               lor_point_1, lor_point_2, 5000);

    for (int timing_num = test_proj_data_info_sptr->get_min_tof_pos_num();
         timing_num <= test_proj_data_info_sptr->get_max_tof_pos_num(); ++ timing_num)
    {
        ProjMatrixElemsForOneBin new_proj_matrix_row;
        Bin bin(seg_num, view_num, axial_num, tang_num, timing_num, 1.f);

        t.reset(); t.start();
        test_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(new_proj_matrix_row,
                                                                          bin);
        t.stop();
        times_of_tofing.push_back(t.value());
//        export_lor(new_proj_matrix_row,
//                   lor_point_1, lor_point_2, timing_num,
//                   proj_matrix_row);
    }

    double mean = 0.0;
    for (unsigned i = 0; i < times_of_tofing.size(); i++)
        mean += times_of_tofing.at(i);

    mean /= (times_of_tofing.size());

    double s=0.0;
    for (unsigned i = 0; i < times_of_tofing.size(); i++)
        s += (times_of_tofing.at(i) - mean) * (times_of_tofing.at(i) - mean) / (times_of_tofing.size()-1);

    s = std::sqrt(s);

    std::cerr<<" Execution  time  for TOF: "<<mean<<" ±"<<s;
    std::cerr << std::endl;
}

void
TOF_Tests::
export_lor(ProjMatrixElemsForOneBin& probabilities,
           const CartesianCoordinate3D<float>& point1,
           const CartesianCoordinate3D<float>& point2,  int current_id)
{
     std::ofstream myfile;
     std::string file_name = "glor_" + boost::lexical_cast<std::string>(current_id) + ".txt";
     myfile.open (file_name.c_str());

     CartesianCoordinate3D<float> voxel_center;

     std::vector<FloatFloat> lor_to_export;
     lor_to_export.reserve(probabilities.size());

     ProjMatrixElemsForOneBin::iterator element_ptr = probabilities.begin();
     while (element_ptr != probabilities.end())
     {
         voxel_center =
                 test_discretised_density_sptr->get_physical_coordinates_for_indices (element_ptr->get_coords());

         if(voxel_center.z() == 0.f)
        {
         project_point_on_a_line(point1, point2, voxel_center );

         float d1 = std::sqrt((point1.x() - voxel_center.x()) *(point1.x() - voxel_center.x()) +
                              (point1.y() - voxel_center.y()) *(point1.y() - voxel_center.y()) +
                              (point1.z() - voxel_center.z()) *(point1.z() - voxel_center.z()));

         float d2 = std::sqrt( (point2.x() - voxel_center.x()) *(point2.x() - voxel_center.x()) +
                               (point2.y() - voxel_center.y()) *(point2.y() - voxel_center.y()) +
                               (point2.z() - voxel_center.z()) *(point2.z() - voxel_center.z()));


         float d12 = (d2 - d1) * 0.5f;

         std::cerr<< voxel_center.x() << " " << voxel_center.y() << " " << voxel_center.z() << " "  <<
                     d1 << " " << d2 << " " << d12 <<std::endl;

         FloatFloat tmp;
         tmp.float1 = d12;
         tmp.float2 = element_ptr->get_value();
         lor_to_export.push_back(tmp);
}
         ++element_ptr;
     }

    for (unsigned int i = 0; i < lor_to_export.size(); i++)
     myfile << lor_to_export.at(i).float1 << "  " << lor_to_export.at(i).float2 << std::endl;

     myfile << std::endl;
     myfile.close();
}

void
TOF_Tests::
export_lor(ProjMatrixElemsForOneBin& probabilities,
           const CartesianCoordinate3D<float>& point1,
           const CartesianCoordinate3D<float>& point2,  int current_id,
           ProjMatrixElemsForOneBin& template_probabilities)
{
     std::ofstream myfile;
     std::string file_name = "glor_" + boost::lexical_cast<std::string>(current_id) + ".txt";
     myfile.open (file_name.c_str());

     CartesianCoordinate3D<float> voxel_center;

     std::vector<FloatFloat> lor_to_export;
     lor_to_export.reserve(template_probabilities.size());

     ProjMatrixElemsForOneBin::iterator tmpl_element_ptr = template_probabilities.begin();
     while (tmpl_element_ptr != template_probabilities.end())
     {
         voxel_center =
                 test_discretised_density_sptr->get_physical_coordinates_for_indices (tmpl_element_ptr->get_coords());
         if(voxel_center.z() == 0.f)
        {
         project_point_on_a_line(point1, point2, voxel_center );

         float d1 = std::sqrt((point1.x() - voxel_center.x()) *(point1.x() - voxel_center.x()) +
                              (point1.y() - voxel_center.y()) *(point1.y() - voxel_center.y()) +
                              (point1.z() - voxel_center.z()) *(point1.z() - voxel_center.z()));

         float d2 = std::sqrt( (point2.x() - voxel_center.x()) *(point2.x() - voxel_center.x()) +
                               (point2.y() - voxel_center.y()) *(point2.y() - voxel_center.y()) +
                               (point2.z() - voxel_center.z()) *(point2.z() - voxel_center.z()));

         float d12 = (d2 - d1) * 0.5f;

         FloatFloat tmp;
         tmp.float1 = d12;

         ProjMatrixElemsForOneBin::iterator element_ptr = probabilities.begin();
         bool found = false;

         while (element_ptr != probabilities.end())
         {
             if (element_ptr->get_coords() == tmpl_element_ptr->get_coords())
             {
                tmp.float2 = element_ptr->get_value();
                found = true;
                break;
             }
             ++element_ptr;
         }

         if (!found)
             tmp.float2 = 0.f;


         lor_to_export.push_back(tmp);
}
         ++tmpl_element_ptr;
     }

    for (unsigned int i = 0; i < lor_to_export.size(); i++)
     myfile << lor_to_export.at(i).float1 << "  " << lor_to_export.at(i).float2 << std::endl;

     myfile << std::endl;
     myfile.close();
}

END_NAMESPACE_STIR

int main()
{
    USING_NAMESPACE_STIR
    TOF_Tests tests;
    tests.run_tests();
    return tests.main_return_value();
}
