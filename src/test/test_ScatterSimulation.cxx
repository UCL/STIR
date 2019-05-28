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
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
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

    void test_scatter_simulation();

    void create_scatter_sinogram();

    void create_attenuation_sinogram(bool ACF = true);
    //! Base function for the scatter simulation tests
//    void simulate_scatter_one_voxel();
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
ScatterSimulationTests::test_scatter_simulation()
{
    shared_ptr<SingleScatterSimulation> sss(new SingleScatterSimulation());

    Scanner::Type type= Scanner::E931;
    shared_ptr<Scanner> test_scanner(new Scanner(type));

    test_scanner->set_reference_energy(511);
    test_scanner->set_energy_resolution(0.11);

    shared_ptr<ExamInfo> exam(new ExamInfo);
    exam->set_low_energy_thres(450);
    exam->set_high_energy_thres(650);
    sss->set_exam_info_sptr(exam);

    // Create the original projdata
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> original_projdata_info( dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(
                                                                        ProjDataInfo::ProjDataInfoCTI(test_scanner,
                                                                                                      1, 0,
                                                                                                      test_scanner->get_num_detectors_per_ring()/2,
                                                                                                      test_scanner->get_max_num_non_arccorrected_bins(),
                                                                                                      false)));

    shared_ptr<VoxelsOnCartesianGrid<float> > tmpl_density( new VoxelsOnCartesianGrid<float>(*original_projdata_info));

    sss->set_template_proj_data_info_sptr(original_projdata_info);
    sss->downsample_scanner(1,8);
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> output_projdata_info(sss->get_template_proj_data_info_sptr());

    shared_ptr<ProjDataInMemory> sss_output(new ProjDataInMemory(exam, output_projdata_info));
    sss->set_output_proj_data_sptr(sss_output);

    shared_ptr<VoxelsOnCartesianGrid<float> > water_density(tmpl_density->clone());
    {
        EllipsoidalCylinder phantom(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
                                    tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.25,
                                    tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.25,
                                    tmpl_density->get_origin());

        CartesianCoordinate3D<int> num_samples(3,3,3);
        phantom.construct_volume(*water_density, num_samples);
        // Water attenuation coefficient.
        *water_density *= 9.687E-02;

    }

    sss->set_density_image_sptr(water_density);
    sss->set_density_image_for_scatter_points_sptr(water_density);
    sss->downsample_image(0.25, 0.5);
    sss->set_random_point(false);

    shared_ptr<VoxelsOnCartesianGrid<float> > act_density(tmpl_density->clone());
    {
        CartesianCoordinate3D<float> centre(tmpl_density->get_origin());
        centre[3] += 80.f;
        EllipsoidalCylinder phantom(tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
                                    tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.0625,
                                    tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.0625,
                                    centre);

        CartesianCoordinate3D<int> num_samples(3,3,3);
        phantom.construct_volume(*act_density, num_samples);
        // Water attenuation coefficient.
    }

    sss->set_activity_image_sptr(act_density);

    check(sss->process_data() == Succeeded::yes ? true : false, "Check Scatter Simulation process");

//    shared_ptr<ProjDataInMemory> atten_sino(new ProjDataInMemory(exam, output_projdata_info));
//    atten_sino->fill(1.F);
//    shared_ptr<ProjDataInMemory> act_sino(new ProjDataInMemory(exam, output_projdata_info));
//    {
//        info("ScatterSimulationTests: Calculate the Attenuation coefficients.");
//        shared_ptr<ForwardProjectorByBin> forw_projector_ptr;
//        shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
//        forw_projector_ptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));

//        shared_ptr<BinNormalisation> normalisation_ptr
//                (new BinNormalisationFromAttenuationImage(water_density,
//                                                          forw_projector_ptr));

//        {
//            normalisation_ptr->set_up(output_projdata_info->create_shared_clone());
//            const double start_frame = 0;
//            const double end_frame = 0;
//            shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_sptr(forw_projector_ptr->get_symmetries_used()->clone());
//            normalisation_ptr->undo(*atten_sino,start_frame,end_frame, symmetries_sptr);
//        }

////        atten_sino->write_to_file("_sino");
////        std::string density_image_for_scatter_points_output_filename("./image");
////        OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
////                write_to_file(density_image_for_scatter_points_output_filename,
////                              *act_density);

//        {
//            forw_projector_ptr->set_up(output_projdata_info->create_shared_clone(), act_density);
//            forw_projector_ptr->forward_project(*act_sino, *act_density);
//        }

//        for (int i = output_projdata_info->get_min_view_num();
//             i < output_projdata_info->get_max_view_num(); ++i)
//        {
//            Viewgram<float> view_att = atten_sino->get_viewgram(i, 0);
//            Viewgram<float> view_act = act_sino->get_viewgram(i,0);
//            Viewgram<float> view_sct = sss_output->get_viewgram(i,0);

//            view_act *= view_att;
//            view_act += view_sct;

//            act_sino->set_viewgram(view_act);
//        }
//    }
//    int nikos = 0;
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

    test_scatter_simulation();
}


END_NAMESPACE_STIR


int main()
{
    USING_NAMESPACE_STIR

    ScatterSimulationTests tests;
    tests.run_tests();
    return tests.main_return_value();
}
