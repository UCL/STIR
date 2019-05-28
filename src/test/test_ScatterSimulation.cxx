//
//
/*
    Copyright (C) 2019, University of Hull
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
  \ingroup test

  \brief Test program for ScatterSimulation

  \author Nikos Efthimiou

*/

#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/scatter/SingleScatterSimulation.h"

#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Box3D.h"
#include "stir/IO/write_to_file.h"
#include <iostream>
#include <math.h>

using std::cerr;
using std::endl;
using std::string;

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for Scanner
*/
class ScatterSimulationTests: public RunTests
{
public:  
    void run_tests();
private:
    //! Load a ProjDataInfo downsample and perform some consistency checks.
    void test_downsampling_ProjDataInfo();
    //! Load an attenuation image for scatter points, downsample and check if
    //! the mean value is approximately the same.
    void test_downsampling_DiscretisedDensity();
    //! Base function for the scatter simulation tests
    void simulate_scatter_one_voxel();
    //! scatter simulation test for one point.
//    void simulate_scatter_for_one_point(shared_ptr<SingleScatterSimulation>);
};


void ScatterSimulationTests::
test_downsampling_ProjDataInfo()
{

    Scanner::Type type= Scanner::E931;
    shared_ptr<Scanner> test_scanner(new Scanner(type));

    // Create the original projdata
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> original_projdata( dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(
                                                                        ProjDataInfo::ProjDataInfoCTI(test_scanner,
                                                                                                      1, test_scanner->get_num_rings()-1,
                                                                                                      test_scanner->get_num_detectors_per_ring()/2,
                                                                                                      test_scanner->get_max_num_non_arccorrected_bins(),
                                                                                                      false)));


    SingleScatterSimulation sss;
    sss.set_template_proj_data_info_sptr(original_projdata);

    {
        shared_ptr<ProjDataInfoCylindricalNoArcCorr> sss_projdata(sss.get_template_proj_data_info_sptr());
        check(*original_projdata == *sss_projdata, "Check the ProjDataInfo has been set correctly.");
    }

    //Even number
    {
        int down_rings = 2;
        int down_dets = 2;
        sss.downsample_scanner(down_rings, down_dets);
        shared_ptr<ProjDataInfoCylindricalNoArcCorr> sss_projdata(sss.get_template_proj_data_info_sptr());
        check_if_equal(original_projdata->get_scanner_ptr()->get_num_rings(), 2*sss_projdata->get_scanner_ptr()->get_num_rings(), "Check the number of rings is correct");
        check_if_equal(original_projdata->get_scanner_ptr()->get_num_detectors_per_ring(),
              2*sss_projdata->get_scanner_ptr()->get_num_detectors_per_ring(), "Check number of detectors per ring.");

        set_tolerance(0.01);
        check_if_equal(2.f*original_projdata->get_ring_spacing(), sss_projdata->get_ring_spacing(), "Check the ring spacing.");
        check_if_equal(2.f*original_projdata->get_axial_sampling(0), sss_projdata->get_axial_sampling(0), "Check axial samping. Seg 0");

        check_if_equal(2.f*original_projdata->get_axial_sampling(original_projdata->get_min_segment_num()),
              sss_projdata->get_axial_sampling(sss_projdata->get_min_segment_num()), "Check axial samping. Min. Seg");
        check_if_equal(2.f*original_projdata->get_axial_sampling(original_projdata->get_max_segment_num()),
              sss_projdata->get_axial_sampling(sss_projdata->get_max_segment_num()), "Check axial samping. Max Seg.");

        Bin b1(original_projdata->get_min_segment_num(),0,
               original_projdata->get_max_axial_pos_num(original_projdata->get_min_segment_num())/2,0);
        Bin b2(sss_projdata->get_min_segment_num(),0,
               sss_projdata->get_max_axial_pos_num(sss_projdata->get_min_segment_num())/2,0);
        check_if_equal(original_projdata->get_m(b1), sss_projdata->get_m(b2), "Check center of Bin (min_seg, 0, mid_axial, 0, 0)");

        check_if_equal(original_projdata->get_scanner_ptr()->get_num_detectors_per_ring(),
              down_dets*sss_projdata->get_scanner_ptr()->get_num_detectors_per_ring(), "Check the number of detectors per ring.");

        check_if_equal(original_projdata->get_num_views(),
              down_dets*sss_projdata->get_num_views(), "Check the number of detectors per ring.");

    }
}

void ScatterSimulationTests::
test_downsampling_DiscretisedDensity()
{
    Scanner::Type type= Scanner::E931;
    shared_ptr<Scanner> test_scanner(new Scanner(type));

    // Create the original projdata
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> original_projdata( dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(
                                                                        ProjDataInfo::ProjDataInfoCTI(test_scanner,
                                                                                                      1, test_scanner->get_num_rings()-1,
                                                                                                      test_scanner->get_num_detectors_per_ring()/2,
                                                                                                      test_scanner->get_max_num_non_arccorrected_bins(),
                                                                                                      false)));

    // Create an appropriate image for the projdata.
    shared_ptr<VoxelsOnCartesianGrid<float> > tmpl_density( new VoxelsOnCartesianGrid<float>(*original_projdata));


    Box3D phantom(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
              tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.25,
              tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.25,
              tmpl_density->get_origin());


    CartesianCoordinate3D<int> num_samples(3,3,3);
    shared_ptr<VoxelsOnCartesianGrid<float> > water_density(tmpl_density->clone());

    phantom.construct_volume(*water_density, num_samples);
    // Water attenuation coefficient.
    *water_density *= 9.687E-02;

//    EllipsoidalCylinder cyl2(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
//                            tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.125,
//                            tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.125,
//                            tmpl_density->get_origin());

//    shared_ptr<VoxelsOnCartesianGrid<float> > bone_density(tmpl_density->clone());

//    cyl2.construct_volume(*bone_density, num_samples);

//    // Watter attenuation coefficient.
//    *bone_density *= 9.687E-02 - 9.696E-02;

    shared_ptr<VoxelsOnCartesianGrid<float> > atten_density(new VoxelsOnCartesianGrid<float>(*tmpl_density));

//    *atten_density = *bone_density;
    *atten_density += *water_density;

    SingleScatterSimulation sss;

    sss.set_template_proj_data_info_sptr(original_projdata);
    sss.set_density_image_for_scatter_points_sptr(atten_density);

//    int total_scatter_points_orig = sss.get_num_scatter_points();

    sss.downsample_image(0.5f, 0.5f, 1);

    shared_ptr<VoxelsOnCartesianGrid<float> > downed_image(
                dynamic_cast<VoxelsOnCartesianGrid<float> *> (sss.get_density_image_for_scatter_points_sptr().get()));

    float mean_value_atten = 0.0f;
    int atten_counter = 0;

    BasicCoordinate<3,int> min_index, max_index ;
    CartesianCoordinate3D<int> coord;

    atten_density->get_regular_range(min_index, max_index);

    for(coord[1]=min_index[1];coord[1]<=max_index[1];++coord[1])
      for(coord[2]=min_index[2];coord[2]<=max_index[2];++coord[2])
        for(coord[3]=min_index[3];coord[3]<=max_index[3];++coord[3])
                if((*atten_density)[coord] > 0.02f)
                {
                    atten_counter++;
                    mean_value_atten += (*atten_density)[coord];
                }

    mean_value_atten /= atten_counter;

    float mean_value_downed = 0.0f;
    int downed_counter = 0;

    downed_image->get_regular_range(min_index, max_index);

    for(coord[1]=min_index[1]+1;coord[1]<=max_index[1]-1;++coord[1])
        for(coord[2]=min_index[2];coord[2]<=max_index[2];++coord[2])
            for(coord[3]=min_index[3];coord[3]<=max_index[3];++coord[3])
                if((*downed_image)[coord] > 0.02f)
                {
                    downed_counter++;
                    mean_value_downed += (*downed_image)[coord];
                }

    mean_value_downed /= downed_counter;

    set_tolerance(0.1);
    check_if_equal(mean_value_atten, mean_value_downed, "Check the mean value of downsampled image.");

//    int total_scatter_points_down = sss.get_num_scatter_points();

   //    std::string density_image_for_scatter_points_output_filename("./nikos");
//    OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
//            write_to_file(density_image_for_scatter_points_output_filename,
//                          *downed_image);

}

void
ScatterSimulationTests::simulate_scatter_one_voxel()
{
    Scanner::Type type= Scanner::E931;
    shared_ptr<Scanner> test_scanner(new Scanner(type));

    test_scanner->set_reference_energy(511);
    test_scanner->set_energy_resolution(0.11);

    shared_ptr<ExamInfo> exam(new ExamInfo);
    exam->set_low_energy_thres(450);
    exam->set_high_energy_thres(650);

    // Create the original projdata
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> original_projdata_info( dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(
                                                                        ProjDataInfo::ProjDataInfoCTI(test_scanner,
                                                                                                      1, 0,
                                                                                                      test_scanner->get_num_detectors_per_ring()/2,
                                                                                                      test_scanner->get_max_num_non_arccorrected_bins(),
                                                                                                      false)));

    shared_ptr<VoxelsOnCartesianGrid<float> > tmpl_density( new VoxelsOnCartesianGrid<float>(*original_projdata_info));

    shared_ptr<SingleScatterSimulation> sss(new SingleScatterSimulation());
    sss->set_template_proj_data_info_sptr(original_projdata_info);
    sss->set_exam_info_sptr(exam);
//    sss->downsample_scanner(1, 4);
    {

    BasicCoordinate<3,int> min_coord = make_coordinate(0,0,0);
    BasicCoordinate<3,int> max_coord = make_coordinate(0,0,0);
    IndexRange<3> range(min_coord,max_coord);

    shared_ptr<VoxelsOnCartesianGrid<float> > one_density( new VoxelsOnCartesianGrid<float>(range, tmpl_density->get_origin(),
                                                                                            tmpl_density->get_grid_spacing()));

    shared_ptr<VoxelsOnCartesianGrid<float> > one_act_density(new VoxelsOnCartesianGrid<float>(*one_density));
    one_act_density->fill(1.0);
    sss->set_activity_image_sptr(one_act_density);
    sss->set_random_point(false);

    shared_ptr<VoxelsOnCartesianGrid<float> > one_att_density(new VoxelsOnCartesianGrid<float>(*one_density));
    one_att_density->fill(9.687E-02);
    sss->set_density_image_sptr(one_att_density);
    sss->set_density_image_for_scatter_points_sptr(one_att_density);

    shared_ptr<ProjDataInMemory> sss_output(new ProjDataInMemory(exam, original_projdata_info));
    sss->set_output_proj_data_sptr(sss_output);

    sss->process_data();

    sss_output->write_to_file("nikos_sino");

//    Box3D phantom(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
//              tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.25,
//              tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.25,
//              tmpl_density->get_origin());


//    CartesianCoordinate3D<int> num_samples(3,3,3);
//    shared_ptr<VoxelsOnCartesianGrid<float> > water_density(tmpl_density->clone());

//    phantom.construct_volume(*water_density, num_samples);
//    // Water attenuation coefficient.
//    *water_density *= 9.687E-02;

    std::string density_image_for_scatter_points_output_filename("./nikos");
     OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
             write_to_file(density_image_for_scatter_points_output_filename,
                           *one_act_density);

//    simulate_scatter_for_one_point(sss);
    }

    int nikos = 0;
}


//void
//ScatterSimulationTests::simulate_scatter_for_one_point(shared_ptr<SingleScatterSimulation>)
//{

//}

void
ScatterSimulationTests::
run_tests()
{

    // test the downsampling functions.
//    test_downsampling_ProjDataInfo();

//    test_downsampling_DiscretisedDensity();

    simulate_scatter();
}


END_NAMESPACE_STIR


int main()
{
    USING_NAMESPACE_STIR

    ScatterSimulationTests tests;
    tests.run_tests();
    return tests.main_return_value();
}
