/*
    Copyright (C) 2016, 2022, UCL
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

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/DetectionPositionPair.h"
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
#ifdef HAVE_CERN_ROOT
#include "stir/listmode/CListRecordROOT.h"
#endif
#include "stir/info.h"
#include "stir/warning.h"
#include <cmath>

START_NAMESPACE_STIR

//! A helper class to keep the combination of a view, a segment and
//! a key tight.
//! \author Nikos Efthimiou
//!
class cache_index{
public:
    cache_index():
    key(0){
        view_num = 0;
        seg_num = 0;
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

/*!
  \ingroup test
  \brief Test class for Time Of Flight
  \author Nikos Efthimiou


  The following tests are performed:

  *. Compare the ProjDataInfo of the GE Signa scanner to known values.

  *. Check if get_det_pos_pair_for_bin swaps detectors (or timing_pos) for bins with opposite timing_pos

  *. Check that the sum of the TOF LOR is the same as the non TOF.

  \warning If you change the mashing factor the test_tof_proj_data_info() will fail.
  \warning The execution time strongly depends on the value of the TOF mashing factor
*/
class TOF_Tests : public RunTests
{
public:
    void run_tests();

private:

    void test_tof_proj_data_info_kernel();
    void test_tof_proj_data_info_det_pos();
    void test_tof_proj_data_info();
#ifdef HAVE_CERN_ROOT
  void test_CListEventROOT();
#endif
    //! This checks peaks a specific bin, finds the LOR and applies all the
    //! kernels of all available timing positions. Then check if the sum
    //! of the TOF bins is equal to the non-TOF LOR.
    void test_tof_kernel_application(bool export_to_file);

    //! Exports the nonTOF LOR to a file indicated by the current_id value
    //! in the filename.
    void
    export_lor(ProjMatrixElemsForOneBin& probabilities,
               const CartesianCoordinate3D<float>& point1,
               const CartesianCoordinate3D<float>& point2,int current_id);

    //! Exports the TOF LOR. The TOFid is indicated in the fileName.
    //! Only the common elements with the nonTOF LOR will be written in the file.
    //! Although changing that is straight forward.
    void
    export_lor(ProjMatrixElemsForOneBin& probabilities,
               const CartesianCoordinate3D<float>& point1,
               const CartesianCoordinate3D<float>& point2,int current_id,
               ProjMatrixElemsForOneBin& template_probabilities);

    shared_ptr<Scanner> test_scanner_sptr;
    shared_ptr<ProjDataInfo> test_proj_data_info_sptr;

    shared_ptr<Scanner> test_nonTOF_scanner_sptr;
    shared_ptr<ProjDataInfo> test_nonTOF_proj_data_info_sptr;

    shared_ptr<DiscretisedDensity<3, float> > test_discretised_density_sptr;
    shared_ptr<ProjMatrixByBin> test_proj_matrix_sptr;
    shared_ptr<ProjMatrixByBin> test_nonTOF_proj_matrix_sptr;
};

void
TOF_Tests::run_tests()
{
    // New Scanner
    test_scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa));
    test_nonTOF_scanner_sptr.reset(new Scanner(Scanner::PETMR_Signa_nonTOF));

    // New Proj_Data_Info
    const int test_tof_mashing_factor = 39; // to have 9 TOF bins (381/39=9)
    test_proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(test_scanner_sptr,
                                                                 1,test_scanner_sptr->get_num_rings() -1,
                                                                 test_scanner_sptr->get_num_detectors_per_ring()/2,
                                                                 test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                                 /* arc_correction*/false));
    test_proj_data_info_sptr->set_tof_mash_factor(test_tof_mashing_factor);

    test_nonTOF_proj_data_info_sptr.reset(ProjDataInfo::ProjDataInfoCTI(test_nonTOF_scanner_sptr,
                                                                 1,test_scanner_sptr->get_num_rings() -1,
                                                                 test_scanner_sptr->get_num_detectors_per_ring()/2,
                                                                 test_scanner_sptr->get_max_num_non_arccorrected_bins(),
                                                                 /* arc_correction*/false));

    test_tof_proj_data_info();
    //    test_tof_geometry_1();

#ifdef HAVE_CERN_ROOT
    test_CListEventROOT();
#endif

    // New Discretised Density
    test_discretised_density_sptr.reset( new VoxelsOnCartesianGrid<float> (*test_proj_data_info_sptr, 1.f,
                                                                           CartesianCoordinate3D<float>(0.f, 0.f, 0.f),
                                                                           CartesianCoordinate3D<int>(-1, -1, -1)));
    // New ProjMatrix
    test_proj_matrix_sptr.reset(new ProjMatrixByBinUsingRayTracing());
    dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_proj_matrix_sptr.get())->set_num_tangential_LORs(1);
    dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_proj_matrix_sptr.get())->set_up(test_proj_data_info_sptr, test_discretised_density_sptr);

    test_nonTOF_proj_matrix_sptr.reset(new ProjMatrixByBinUsingRayTracing());
    dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_nonTOF_proj_matrix_sptr.get())->set_num_tangential_LORs(1);
    dynamic_cast<ProjMatrixByBinUsingRayTracing*>(test_nonTOF_proj_matrix_sptr.get())->set_up(test_nonTOF_proj_data_info_sptr, test_discretised_density_sptr);

    // Switch to true in order to export the LORs at files in the current directory
    test_tof_kernel_application(false);
}

void
TOF_Tests::test_tof_proj_data_info_kernel()
{
    const int correct_tof_mashing_factor = 39;
    const int num_timing_positions = 9;
    float correct_width_of_tof_bin = test_scanner_sptr->get_size_of_timing_pos() *
            test_proj_data_info_sptr->get_tof_mash_factor() * 0.299792458f/2;
    float correct_timing_locations[num_timing_positions] = {-360.201f/2 + correct_width_of_tof_bin/2, -280.156f/2 + correct_width_of_tof_bin/2,
                                                            -200.111f/2 + correct_width_of_tof_bin/2, -120.067f/2 + correct_width_of_tof_bin/2,
                                                            0.0f, 40.022f/2 + correct_width_of_tof_bin/2,
                                                            120.067f/2 + correct_width_of_tof_bin/2, 200.111f/2 + correct_width_of_tof_bin/2,
                                                            280.156f/2+ correct_width_of_tof_bin/2};

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
TOF_Tests::test_tof_proj_data_info_det_pos()
{

  auto pdi_ptr =
    dynamic_cast<ProjDataInfoCylindricalNoArcCorr const*> (test_proj_data_info_sptr.get());

  Bin b1(1,2,3,4,5);
  Bin b2 = b1;
  b2.timing_pos_num() = -b1.timing_pos_num();

  DetectionPositionPair<> dp1, dp2;
  pdi_ptr->get_det_pos_pair_for_bin(dp1, b1);
  pdi_ptr->get_det_pos_pair_for_bin(dp2, b2);

  check((dp1.timing_pos() == dp2.timing_pos() && dp1.pos1() == dp2.pos2() && dp1.pos2() == dp2.pos1())
        || (static_cast<int>(dp1.timing_pos()) == -static_cast<int>(dp2.timing_pos()) && dp1.pos1() == dp2.pos1() && dp1.pos2() == dp2.pos2()),
        "get_det_pos_for_bin with bins of opposite timing_pos");
}


void
TOF_Tests::test_tof_proj_data_info()
{
  test_tof_proj_data_info_kernel();
  test_tof_proj_data_info_det_pos();
}

#ifdef HAVE_CERN_ROOT
void TOF_Tests::test_CListEventROOT()
{
  std::cerr << "CListEventROOT tests\n";
  const auto old_tol = this->get_tolerance();
  // set tolerance to ~1mm. It has to be surprisingly large at the moment. Problem in the LOR functions? (TODO)
  this->set_tolerance(3.F);

  test_proj_data_info_sptr->set_tof_mash_factor(1);

  const int ring1 = 1, ring2 = 4, crystal1 = 0, crystal2 = 25;
  const float delta_time = 800.F;

  CListEventROOT event(test_proj_data_info_sptr);
  event.init_from_data(ring1, ring2, crystal1, crystal2, delta_time);
  Bin bin;
  // this doesn't set time_frame, so force that to 1 for later comparisons
  bin.time_frame_num() = 1;

  event.get_bin(bin, *test_proj_data_info_sptr);
  check(bin.timing_pos_num() != 0, "test CListEventROOT non-zero TOF bin");

  DetectionPositionPair<> det_pos;
  event.get_detection_position(det_pos);

  LORAs2Points<float> lor_2pts(event.get_LOR());
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_sc;
  test_proj_data_info_sptr->get_LOR(lor_sc, bin);
  LORAs2Points<float> test_lor(lor_sc);
  check_if_equal(lor_2pts.p1(), test_lor.p1(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 1");
  check_if_equal(lor_2pts.p2(), test_lor.p2(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 2");

  event.init_from_data(ring2, ring1, crystal2, crystal1, -delta_time);
  {
    Bin bin_swapped;
    bin_swapped.time_frame_num() = 1;
    event.get_bin(bin_swapped, *test_proj_data_info_sptr);
    check_if_equal(bin_swapped, bin, "CListEventROOT: get_bin with swapped detectors");
    {
      DetectionPositionPair<> det_pos_swapped;
      event.get_detection_position(det_pos_swapped);
      if (det_pos_swapped.timing_pos() == det_pos.timing_pos())
        {
          check_if_equal(det_pos_swapped.pos1(), det_pos.pos1(), "CListEventROOT: get_detection_position with swapped detectors: equal timing_pos, but different pos1");
          check_if_equal(det_pos_swapped.pos2(), det_pos.pos2(), "CListEventROOT: get_detection_position with swapped detectors: equal timing_pos, but different pos2");
        }
      else if (det_pos_swapped.timing_pos() == event.get_scanner_ptr()->get_max_num_timing_poss()-1-det_pos.timing_pos())
        {
          check_if_equal(det_pos_swapped.pos2(), det_pos.pos1(), "CListEventROOT: get_detection_position with swapped detectors: opposite timing_pos, but different pos1/2");
          check_if_equal(det_pos_swapped.pos1(), det_pos.pos2(), "CListEventROOT: get_detection_position with swapped detectors: opposite timing_pos, but different pos1/2");
        }
      else
        {
          check_if_equal(std::abs(det_pos_swapped.timing_pos()), std::abs(det_pos.timing_pos()),
                         "CListEventROOT: get_detection_position with swapped detectors: wrong timing_pos");
        }
    }

    LORAs2Points<float> lor_2pts_swapped(event.get_LOR());
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor_sc_swapped;
    test_proj_data_info_sptr->get_LOR(lor_sc_swapped, bin_swapped);
    LORAs2Points<float> test_lor_swapped(lor_sc);
    check_if_equal(lor_2pts_swapped.p1(), test_lor_swapped.p1(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 3");
    check_if_equal(lor_2pts_swapped.p2(), test_lor_swapped.p2(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 4");

    // now check if equal
    check_if_equal(bin, bin_swapped, "CListEventROOT:get_bin for reordered detectors");
    check_if_equal(lor_2pts_swapped.p1(), lor_2pts.p1(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 5");
    check_if_equal(lor_2pts_swapped.p2(), lor_2pts.p2(), "CListEventROOT::get_LOR and ProjDataInfo::get_LOR consistency check 6");
  }
  // repeat with swapped detectors
  this->set_tolerance(old_tol);
}
#endif

void
TOF_Tests::test_tof_kernel_application(bool print_to_file)
{
    int seg_num = 0;
    int view_num = 0;
    int axial_num = 0;
    int tang_num = 0;

    float nonTOF_val = 0.0;
    float TOF_val = 0.0;

    ProjMatrixElemsForOneBin proj_matrix_row;
    ProjMatrixElemsForOneBin sum_tof_proj_matrix_row;

    HighResWallClockTimer t;
    std::vector<double> times_of_tofing;

    ProjDataInfoCylindrical* proj_data_ptr =
            dynamic_cast<ProjDataInfoCylindrical*> (test_proj_data_info_sptr.get());

//    ProjDataInfoCylindrical* proj_data_nonTOF_ptr =
//            dynamic_cast<ProjDataInfoCylindrical*> (test_nonTOF_proj_data_info_sptr.get());

    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;

    Bin this_bin(seg_num, view_num, axial_num, tang_num, 1.f);

    t.reset(); t.start();
    test_nonTOF_proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, this_bin);
    t.stop();

    std::cerr<<"Execution time for nonTOF: "<<t.value() << std::endl;
    proj_data_ptr->get_LOR(lor, this_bin);
    LORAs2Points<float> lor2(lor);

    if (print_to_file)
        export_lor(proj_matrix_row,
                   lor2.p1(), lor2.p2(), 500000000);

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

        if (print_to_file)
            export_lor(new_proj_matrix_row,
                       lor2.p1(), lor2.p2(), timing_num,
                       proj_matrix_row);


        if (sum_tof_proj_matrix_row.size() > 0)
        {
            ProjMatrixElemsForOneBin::iterator element_ptr = new_proj_matrix_row.begin();
            while (element_ptr != new_proj_matrix_row.end())
            {
                ProjMatrixElemsForOneBin::iterator sum_element_ptr = sum_tof_proj_matrix_row.begin();
                bool found = false;
                while(sum_element_ptr != sum_tof_proj_matrix_row.end())
                {
                    if(element_ptr->get_coords() == sum_element_ptr->get_coords())
                    {
                        float new_value = element_ptr->get_value() + sum_element_ptr->get_value();
                        *sum_element_ptr = ProjMatrixElemsForOneBin::value_type(element_ptr->get_coords(), new_value);
                        found = true;
                        break;
                    }
                    ++sum_element_ptr;
                }
                if (!found)
                {
                    sum_tof_proj_matrix_row.push_back(ProjMatrixElemsForOneBin::value_type(element_ptr->get_coords(),
                                                                                           element_ptr->get_value()));
                    break;
                }
                ++element_ptr;
            }

        }
        else
        {
            ProjMatrixElemsForOneBin::iterator element_ptr = new_proj_matrix_row.begin();
            while (element_ptr != new_proj_matrix_row.end())
            {
                sum_tof_proj_matrix_row.push_back(ProjMatrixElemsForOneBin::value_type(element_ptr->get_coords(),
                                                                                       element_ptr->get_value()));
                ++element_ptr;
            }
        }

    }

    // Get value of nonTOF LOR, for central voxels only

    {
        ProjMatrixElemsForOneBin::iterator element_ptr = proj_matrix_row.begin();
        while (element_ptr != proj_matrix_row.end())
        {
            if (element_ptr->get_value() > nonTOF_val)
                nonTOF_val = element_ptr->get_value();
            ++element_ptr;
        }
    }

    // Get value of TOF LOR, for central voxels only

    {
        ProjMatrixElemsForOneBin::iterator element_ptr = sum_tof_proj_matrix_row.begin();
        while (element_ptr != sum_tof_proj_matrix_row.end())
        {
            if (element_ptr->get_value() > TOF_val)
                TOF_val = element_ptr->get_value();
            ++element_ptr;
        }
    }


    check_if_equal(static_cast<double>(nonTOF_val), static_cast<double>(TOF_val),
                   "Sum over nonTOF LOR does not match sum over TOF LOR.");

    {
        double mean = 0.0;
        for (unsigned i = 0; i < times_of_tofing.size(); i++)
            mean += times_of_tofing.at(i);

        mean /= (times_of_tofing.size());

        double s=0.0;
        for (unsigned i = 0; i < times_of_tofing.size(); i++)
            s += (times_of_tofing.at(i) - mean) * (times_of_tofing.at(i) - mean) / (times_of_tofing.size()-1);

        s = std::sqrt(s);
        std::cerr<<"Execution  time  for TOF: "<<mean<<" ±"<<s;
    }

    std::cerr << std::endl;
}

void
TOF_Tests::
export_lor(ProjMatrixElemsForOneBin& probabilities,
           const CartesianCoordinate3D<float>& point1,
           const CartesianCoordinate3D<float>& point2,
           int current_id)
{
    std::ofstream myfile;
    std::string file_name = "glor_" + boost::lexical_cast<std::string>(current_id) + ".txt";
    myfile.open (file_name.c_str());

    CartesianCoordinate3D<float> voxel_center;

    std::vector<FloatFloat> lor_to_export;
    lor_to_export.reserve(probabilities.size());

    const CartesianCoordinate3D<float> middle = (point1 + point2)*0.5f;
    const CartesianCoordinate3D<float> diff = point2 - middle;

    const float lor_length = 1.f / (std::sqrt(diff.x() * diff.x() +
                                        diff.y() * diff.y() +
                                        diff.z() * diff.z()));

    ProjMatrixElemsForOneBin::iterator element_ptr = probabilities.begin();
    while (element_ptr != probabilities.end())
    {
        voxel_center =
                test_discretised_density_sptr->get_physical_coordinates_for_indices (element_ptr->get_coords());

//        if(voxel_center.z() == 0.f)
        {
            project_point_on_a_line(point1, point2, voxel_center );

            const CartesianCoordinate3D<float> x = voxel_center - middle;

            const float d2 = -inner_product(x, diff) * lor_length;

            FloatFloat tmp;
            tmp.float1 = d2;

//            std::cerr<< voxel_center.x() << " " << voxel_center.y() << " " << voxel_center.z() << " "  <<
//                        d1 << " " << d2 << " " << d12 << " " << element_ptr->get_value() <<std::endl;

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

    const CartesianCoordinate3D<float> middle = (point1 + point2)*0.5f;
    const CartesianCoordinate3D<float> diff = point2 - middle;

    const float lor_length = 1.f / (std::sqrt(diff.x() * diff.x() +
                                        diff.y() * diff.y() +
                                        diff.z() * diff.z()));

    CartesianCoordinate3D<float> voxel_center;

    std::vector<FloatFloat> lor_to_export;
    lor_to_export.reserve(template_probabilities.size());

    ProjMatrixElemsForOneBin::iterator tmpl_element_ptr = template_probabilities.begin();
    while (tmpl_element_ptr != template_probabilities.end())
    {
        voxel_center =
                test_discretised_density_sptr->get_physical_coordinates_for_indices (tmpl_element_ptr->get_coords());
//        if(voxel_center.z() == 0.f)
        {
            project_point_on_a_line(point1, point2, voxel_center );

            const CartesianCoordinate3D<float> x = voxel_center - middle;

            const float d2 = -inner_product(x, diff) * lor_length;

            FloatFloat tmp;
            tmp.float1 = d2;

            ProjMatrixElemsForOneBin::iterator element_ptr = probabilities.begin();

            while (element_ptr != probabilities.end())
            {
                if (element_ptr->get_coords() == tmpl_element_ptr->get_coords())
                {
                    tmp.float2 = element_ptr->get_value();
                    lor_to_export.push_back(tmp);
                    break;
                }
                ++element_ptr;
            }
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
