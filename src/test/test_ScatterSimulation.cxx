//
//
/*
    Copyright (C) 2019, University of Hull
    Copyright (C) 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup test

  \brief Test program for stir::ScatterSimulation

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#include "stir/RunTests.h"
#include "stir/Verbosity.h"
#include "stir/Scanner.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/scatter/SingleScatterSimulation.h"
#include "stir/zoom.h"
#include "stir/round.h"
#if 0
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#endif
#include "stir/Shape/EllipsoidalCylinder.h"
#include "stir/Shape/Box3D.h"
#include "stir/IO/write_to_file.h"
#include "stir/stream.h"
#include <iostream>
#include <math.h>
#include "stir/centre_of_gravity.h"

using std::cerr;
using std::endl;
using std::string;

START_NAMESPACE_STIR

/*!
  \ingroup test
  \ingroup scatter
  \brief Test class for ScatterSimulation
*/
class ScatterSimulationTests: public RunTests
{
public:  
    bool write_output;
    void run_tests();
private:

    //! Load a ProjDataInfo downsample and perform some consistency checks.
    void test_downsampling_ProjDataInfo();
    //! Load an attenuation image for scatter points, downsample and check if
    //! the mean value is approximately the same.
    void test_downsampling_DiscretisedDensity();

    //! Do simulation of object in the centre, check if symmetric
    void test_scatter_simulation();

    void test_symmetric(ScatterSimulation& sss, const std::string& name);
    void test_output_is_symmetric(const ProjData& proj_data, const std::string& name);
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


    unique_ptr<SingleScatterSimulation> sss(new SingleScatterSimulation());
    sss->set_template_proj_data_info(*original_projdata);

    {
        auto sss_projdata(sss->get_template_proj_data_info_sptr());
        check(*original_projdata == *sss_projdata, "Check the ProjDataInfo has been set correctly.");
    }

    // Downsample the scanner 50%
    {
        int down_rings = static_cast<int>(test_scanner->get_num_rings()*0.5 + 0.5);
        int down_dets = static_cast<int>(test_scanner->get_num_detectors_per_ring() * 0.5);

        sss->downsample_scanner(down_rings, down_dets);
        auto sss_projdata(sss->get_template_proj_data_info_sptr());
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


    Box3D phantom(tmpl_density->get_x_size()*tmpl_density->get_voxel_size().x()*0.25,
              tmpl_density->get_y_size()*tmpl_density->get_voxel_size().y()*0.25,
              tmpl_density->get_z_size()*tmpl_density->get_voxel_size().z()*2,
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

    unique_ptr<SingleScatterSimulation> sss(new SingleScatterSimulation());

    sss->set_template_proj_data_info(*original_projdata);
    sss->set_density_image_sptr(atten_density);

//    int total_scatter_points_orig = sss.get_num_scatter_points();

    sss->downsample_density_image_for_scatter_points(0.5f, 0.5f, 1);
    
    auto downed_image = sss->get_density_image_for_scatter_points_sptr();


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
    set_tolerance(0.01);

    CartesianCoordinate3D<float> cog_atten = find_centre_of_gravity_in_mm(*atten_density);
    CartesianCoordinate3D<float> cog_downed = find_centre_of_gravity_in_mm(*dynamic_cast<const VoxelsOnCartesianGrid<float>*>(downed_image.get()));


    check_if_equal(cog_atten, cog_downed, "Check centre of gravity of the original image is the same as the downsampled.");
//    int total_scatter_points_down = sss.get_num_scatter_points();

//       std::string density_image_for_scatter_points_output_filename("./output_image");
//       write_to_file(density_image_for_scatter_points_output_filename,
//                          *downed_image);
//int debug_stop = 0;
}

void
ScatterSimulationTests::test_symmetric(ScatterSimulation& ss, const std::string& name)
{
  auto output_projdata_info(ss.get_template_proj_data_info_sptr());
  shared_ptr<ProjDataInMemory> sss_output(new ProjDataInMemory(ss.get_exam_info_sptr(), output_projdata_info));
  ss.set_output_proj_data_sptr(sss_output);

  std::cerr << "\nSetting up for test " << name << "\n";
  check(ss.set_up() == Succeeded::yes ? true : false, "Check Scatter Simulation set_up. test " + name);

  if (this->write_output)
    {
      write_to_file("my_sss_activity__"+name+".hv", ss.get_activity_image());
      write_to_file("my_sss_attenuation_"+name+".hv", ss.get_attenuation_image());
      write_to_file("my_sss_scatter-points_"+name+".hv", ss.get_attenuation_image_for_scatter_points());
    }

  std::cerr << "Starting processing\n";
  check(ss.process_data() == Succeeded::yes ? true : false, "Check Scatter Simulation process. test " + name);

  std::cerr << "Performing checks\n";

  // check if max within 5%
  const SegmentByView<float> seg = sss_output->get_segment_by_view(0);
  //check(seg.find_max()>0.F, "Check Scatter Simulation output not zero. test " + name);
  {
    const float approx_max = 0.195F;
    const double old_tolerance = get_tolerance();
    set_tolerance(.5);
    check_if_equal(seg.find_max()*1000, approx_max*1000,
                   "Check Scatter Simulation output maximum value is approximately ok. test " + name);
    set_tolerance(old_tolerance);
  }

  test_output_is_symmetric(*ss.get_output_proj_data_sptr(), name);

  if (this->write_output)
    sss_output->write_to_file("my_single_scatter_sim_" + name + ".hs");
}

void
ScatterSimulationTests::test_output_is_symmetric(const ProjData& proj_data, const std::string& name)
{
  SegmentBySinogram<float> seg = proj_data.get_segment_by_sinogram(0);
  seg *=1000; // work-around problem in RunTests::check_if_equal for floats that it can fail for small numbers

  // values have to be symmetric around the middle of the scanner
  for (int first=seg.get_min_axial_pos_num(), last=seg.get_max_axial_pos_num();
       first<last;
       ++first, --last)
    check_if_equal(seg[first][0][0], seg[last][0][0], "check if symmetric along the scanner axis. test " + name);

  for (int first=-1, last=+1;
       first>=seg.get_min_tangential_pos_num() && last<=seg.get_max_tangential_pos_num();
       --first, ++last)
    check_if_equal(seg[3][0][first], seg[3][0][last], "check if symmetric along the tangential_pos direction. test " + name);


  // test views. Need to reduce tolerance due to discretisation artefacts
  {
    const double old_tolerance = get_tolerance();
    set_tolerance(.1);
    const int mid_axial_pos_num = (seg.get_min_axial_pos_num()+seg.get_max_axial_pos_num())/2;
    const int first_view = seg.get_min_view_num();
    // we will compare the middle 11 entries
    // (the profile goes down a lot, and we want to avoid errors for very small numbers)
    Array<1,float> row_first = seg[mid_axial_pos_num][first_view];
    row_first.resize(-5,5);
    for (int view=seg.get_min_view_num() + 1; view<=seg.get_max_view_num(); ++view)
      {
        Array<1,float> row = seg[mid_axial_pos_num][view];
        row.resize(-5,5);
        check_if_equal(row_first, row,
                     "check if symmetric along views. test " + name);
      }
    set_tolerance(old_tolerance);
  }
#if 0
  for (int a=seg.get_min_axial_pos_num(), a<=seg.get_max_axial_pos_num(); ++a)
    std::cout<< seg[a][0][0] << ",";
  std::cout<<std::endl;
#endif
}

void
ScatterSimulationTests::test_scatter_simulation()
{
    unique_ptr<SingleScatterSimulation> sss(new SingleScatterSimulation());

    Scanner::Type type= Scanner::E931;
    shared_ptr<Scanner> test_scanner(new Scanner(type));
    const float scanner_length = test_scanner->get_num_rings() * test_scanner->get_ring_spacing();

    std::cerr << "Testing scatter simulation for the following scanner:\n"
              << test_scanner->parameter_info()
              << "\nAxial length = " << scanner_length << " mm" << std::endl;

    if(!test_scanner->has_energy_information())
    {
        test_scanner->set_reference_energy(511);
        test_scanner->set_energy_resolution(0.34f);
    }

    check(test_scanner->has_energy_information() == true, "Check the scanner has energy information.");

    shared_ptr<ExamInfo> exam(new ExamInfo);
    exam->set_low_energy_thres(450);
    exam->set_high_energy_thres(650);
    exam->imaging_modality = ImagingModality::PT;

    check(exam->has_energy_information() == true, "Check the ExamInfo has energy information.");

    sss->set_exam_info(*exam);

    // Create the original projdata
    shared_ptr<ProjDataInfoCylindricalNoArcCorr> original_projdata_info( dynamic_cast<ProjDataInfoCylindricalNoArcCorr* >(
                                                                             ProjDataInfo::ProjDataInfoCTI(test_scanner,
                                                                                                           1, 0,
                                                                                                           test_scanner->get_num_detectors_per_ring()/2,
                                                                                                           test_scanner->get_max_num_non_arccorrected_bins(),
                                                                                                           false)));

    check(original_projdata_info->has_energy_information() == true, "Check the ProjDataInfo has energy information.");

    shared_ptr<VoxelsOnCartesianGrid<float> > tmpl_density( new VoxelsOnCartesianGrid<float>(exam, *original_projdata_info));

    //// Create an object in the middle of the image (which will be in the middle of the scanner
    CartesianCoordinate3D<int> min_ind, max_ind;
    tmpl_density->get_regular_range(min_ind, max_ind);
    CartesianCoordinate3D<float> centre((tmpl_density->get_physical_coordinates_for_indices(min_ind) +
                                         tmpl_density->get_physical_coordinates_for_indices(max_ind))/2.F);

    EllipsoidalCylinder phantom(50.F, 50.F, 50.F, centre);
    CartesianCoordinate3D<int> num_samples(2,2,2);

    //// attenuation image
    shared_ptr<VoxelsOnCartesianGrid<float> > water_density(tmpl_density->clone());
    phantom.construct_volume(*water_density, num_samples);
    // Water attenuation coefficient.
    *water_density *= 9.687E-02;
    sss->set_density_image_sptr(water_density);

    ////activity image (same object)
    shared_ptr<VoxelsOnCartesianGrid<float> > act_density(tmpl_density->clone());
    phantom.construct_volume(*act_density, num_samples);
    sss->set_activity_image_sptr(act_density);

    //// sss settings
    sss->set_randomly_place_scatter_points(false);

    sss->set_template_proj_data_info(*original_projdata_info);
    sss->downsample_scanner(original_projdata_info->get_scanner_sptr()->get_num_rings(), -1);
#if 1
    set_tolerance(.02);
    {
      const int new_size_z = 14; // original_projdata_info->get_scanner_sptr()->get_num_rings()
      const float zoom_z = static_cast<float>(new_size_z-1)/(water_density->size()-1);
      sss->downsample_density_image_for_scatter_points(.2F, zoom_z, -1, new_size_z);
      test_symmetric(*sss, "rings_size14");
    }
    {
      const int new_size_z = 14; // original_projdata_info->get_scanner_sptr()->get_num_rings()
      sss->downsample_density_image_for_scatter_points(.2F, -1.F, -1, new_size_z);
      test_symmetric(*sss, "rings_size14_def_zoom");
    }
    {
      sss->downsample_density_image_for_scatter_points(.3F, .4F, -1, -1);
      test_symmetric(*sss, "rings_zoomxy.3_zoomz.4");
    }
    // reduce for smaller number of rings
    set_tolerance(.03);
    {
      const int new_size_z = 5;
      sss->downsample_density_image_for_scatter_points(.2F, -1.F, -1, new_size_z);
      test_symmetric(*sss, "rings_size5");
    }
    {
      sss->downsample_density_image_for_scatter_points(.2F, .3F, -1, -1);
      test_symmetric(*sss, "rings_zoomz.3");
    }

    // testing with zooming (but currently fails)
    {
      const CartesianCoordinate3D<float> zooms(.5F, .3F, .3F);
      shared_ptr<VoxelsOnCartesianGrid<float> >
        zoomed_act_sptr(new VoxelsOnCartesianGrid<float>
                        (zoom_image(*act_density,
                                    zooms,
                                    CartesianCoordinate3D<float>(0.F,0.F,0.F) /* offset */,
                                    round(BasicCoordinate<3,float>(act_density->get_lengths())*zooms)+1 /* sizes */,
                                    ZoomOptions::preserve_projections)
                         )
                        );
      //std::cerr << "new origin : " << zoomed_act_sptr->get_origin();
      sss->set_activity_image_sptr(zoomed_act_sptr);
      std::cerr << "\nThis test should currently throw an error. You'll see some error messages therefore.\n";
      try
        {
          test_symmetric(*sss, "act_zoom_rings_zoomxy.3_zoomz.4");
          check(false, "Test on zooming of activity image should have thrown.");
        }
      catch(...)
        {
          // ok
        }
      // restore to original activity
      sss->set_activity_image_sptr(act_density);
    }

    // a few tests with more downsampled scanner
    sss->set_template_proj_data_info(*original_projdata_info);
    sss->downsample_scanner(original_projdata_info->get_scanner_sptr()->get_num_rings()/2, -1);
    {
      const int new_size_z = 14;
      const float zoom_z = static_cast<float>(new_size_z-1)/(water_density->size()-1);
      sss->downsample_density_image_for_scatter_points(.2F, zoom_z, -1, new_size_z);
      test_symmetric(*sss, "halfrings_size14");
    }
    // reduce for smaller number of rings
    set_tolerance(.03);
    {
      const int new_size_z = 5;
      sss->downsample_density_image_for_scatter_points(.2F, -1.F, -1, new_size_z);
      test_symmetric(*sss, "halfrings_size5");
    }
    {
      sss->downsample_density_image_for_scatter_points(.2F, .3F, -1, -1);
      test_symmetric(*sss, "halfrings_zoomz.3");
    }
#endif

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
    ////        write_to_file(density_image_for_scatter_points_output_filename,
    ////                      *act_density);

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
    test_downsampling_ProjDataInfo();
    test_downsampling_DiscretisedDensity();

    test_scatter_simulation();
}


END_NAMESPACE_STIR


int main()
{
    USING_NAMESPACE_STIR
    // decrease verbosity a bit to avoid too much output
    Verbosity::set(0);

    ScatterSimulationTests tests;
    tests.write_output = true; // TODO get this from the command line args
    tests.run_tests();
    return tests.main_return_value();
}
