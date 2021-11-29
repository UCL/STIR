/*!

  \file
  \ingroup recontest

  \brief Test program for back projection and forward projection using stir::ProjDataInfoBlockOnCylindrical

  \author Daniel Deidda

*/
/*  Copyright (C) 2021, National Physical Laboratory
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

#include "stir/info.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ExamInfo.h"
#include "stir/LORCoordinates.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/copy_fill.h"
#include "stir/IndexRange3D.h"
#include "stir/CPUTimer.h"
#include "stir/Shape/Shape3DWithOrientation.h"
#include "stir/Shape/Ellipsoid.h"
#include "stir/Shape/Box3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/IO/write_to_file.h"
//#include "stir/Shape/Shape3D.h"

START_NAMESPACE_STIR


/*!
  \ingroup test
  \brief Test class for Blocks 
*/
class BlocksTests: public RunTests
{
public:
  void run_tests();
private:
  void run_symmetry_test();
  void run_plane_symmetry_test();
};

/*! The following is a test for symmetries: a simulated image is created with a plane at known angles,
 *  the forward projected sinogram should show the makimum value at the bin corresponding to the angl phi 
 *  equal to the orientation of the plane
*/
void
BlocksTests::run_plane_symmetry_test(){
    
    CartesianCoordinate3D<float> origin (0,0,0);  
    CartesianCoordinate3D<float> grid_spacing (1.1,2.2,2.2); 
    float phi1;
    float phi2;
    const IndexRange<3> 
      range(Coordinate3D<int>(0,-45,-44),
            Coordinate3D<int>(24,44,45));
    VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);
    
//    60 degrees
    phi1= 60*_PI/180;
    const Array<2,float> direction_vectors=
  make_array(make_1d_array(1.F,0.F,0.F),
         make_1d_array(0.F,cos(float(_PI)-phi1),sin(float(_PI)-phi1)),
         make_1d_array(0.F,-sin(float(_PI)-phi1),cos(float(_PI)-phi1)));

    Ellipsoid
      plane(CartesianCoordinate3D<float>(/*edge_z*/25*grid_spacing.z(),
                                             /*edge_y*/91*grid_spacing.y(),
                                             /*edge_x*/5*grid_spacing.x()),
		        /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(),
                                                       0*grid_spacing.y(),
                                                       0),
                direction_vectors);
    
    plane.construct_volume(image, make_coordinate(3,3,3));
    
    
//    rotate by 30 degrees
    phi2=30*_PI/180;
    VoxelsOnCartesianGrid<float>  image2=*image.get_empty_copy();
    const Array<2,float> direction2=make_array(make_1d_array(1.F,0.F,0.F),
    make_1d_array(0.F,cos(float(_PI)-phi2),sin(float(_PI)-phi2)),
    make_1d_array(0.F,-sin(float(_PI)-phi2),cos(float(_PI)-phi2)));
    
    Ellipsoid
      plane2(CartesianCoordinate3D<float>(/*edge_z*/25*grid_spacing.z(),
                                             /*edge_y*/91*grid_spacing.y(),
                                             /*edge_x*/5*grid_spacing.x()),
		        /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(),
                                                       0*grid_spacing.y(),
                                                       0),
                direction2);
//    plane.set_direction_vectors(direction2);
        
    plane2.construct_volume(image2, make_coordinate(3,3,3));
    
//    create projadata info
    
    shared_ptr<Scanner> scannerBlocks_ptr;
    scannerBlocks_ptr.reset(new Scanner (Scanner::SAFIRDualRingPrototype));
    scannerBlocks_ptr->set_num_axial_crystals_per_block(1);
    scannerBlocks_ptr->set_axial_block_spacing(scannerBlocks_ptr->get_axial_crystal_spacing()*
                                               scannerBlocks_ptr->get_num_axial_crystals_per_block());
    scannerBlocks_ptr->set_transaxial_block_spacing(scannerBlocks_ptr->get_transaxial_crystal_spacing()*
                                                    scannerBlocks_ptr->get_num_transaxial_crystals_per_block());
//    scannerBlocks_ptr->set_num_transaxial_crystals_per_block(1);
    scannerBlocks_ptr->set_num_axial_blocks_per_bucket(1);
//    scannerBlocks_ptr->set_num_transaxial_blocks_per_bucket(1);
    scannerBlocks_ptr->set_num_rings(1);
    
    scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
    
    
    VectorWithOffset<int> num_axial_pos_per_segment(scannerBlocks_ptr->get_num_rings()*2-1);
    VectorWithOffset<int> min_ring_diff_v(scannerBlocks_ptr->get_num_rings()*2-1);
    VectorWithOffset<int> max_ring_diff_v(scannerBlocks_ptr->get_num_rings()*2-1);
    
    for (int i=0; i<2*scannerBlocks_ptr->get_num_rings()-1; i++){
        min_ring_diff_v[i]=-scannerBlocks_ptr->get_num_rings()+1+i;
        max_ring_diff_v[i]=-scannerBlocks_ptr->get_num_rings()+1+i;
        if (i<scannerBlocks_ptr->get_num_rings())
            num_axial_pos_per_segment[i]=i+1;
        else
            num_axial_pos_per_segment[i]=2*scannerBlocks_ptr->get_num_rings()-i-1;
        }
    
    shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_blocks_ptr;
    proj_data_info_blocks_ptr.reset(
                new ProjDataInfoBlocksOnCylindricalNoArcCorr(
                    scannerBlocks_ptr,
                    num_axial_pos_per_segment,
                    min_ring_diff_v, max_ring_diff_v,
                    scannerBlocks_ptr->get_max_num_views(),
                    scannerBlocks_ptr->get_max_num_non_arccorrected_bins()));
//    now forward-project image
    
    shared_ptr<ForwardProjectorByBin> forw_projector_sptr,forw_projector2_sptr;
    shared_ptr<ProjData> projdata, projdata2;
    shared_ptr<DiscretisedDensity<3,float> > image_sptr(image.clone());
    write_to_file("plane60",*image_sptr);
    
    shared_ptr<DiscretisedDensity<3,float> > image2_sptr(image2.clone());
    write_to_file("plane30",*image2_sptr);
    
    shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
    forw_projector_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    info(boost::format("Test blocks on Cylindrical: Forward projector used: %1%") % forw_projector_sptr->parameter_info());
    
    forw_projector_sptr->set_up(proj_data_info_blocks_ptr,
                                image_sptr);
    
    
    forw_projector2_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    forw_projector2_sptr->set_up(proj_data_info_blocks_ptr,
                                 image2_sptr);

    //-- ExamInfo
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo());
    exam_info_sptr->imaging_modality = ImagingModality::PT;
    
    projdata.reset(new ProjDataInterfile(exam_info_sptr,
                                         proj_data_info_blocks_ptr,
                                         "sino1_from_plane.hs",
                                         std::ios::out | std::ios::trunc | std::ios::in));

    
    forw_projector_sptr->forward_project(*projdata, *image_sptr);
    
    
    projdata2.reset(new ProjDataInterfile(exam_info_sptr,
                                          proj_data_info_blocks_ptr,
                                          "sino2_from_plane.hs",
                                          std::ios::out | std::ios::trunc | std::ios::in));
     
    
    forw_projector2_sptr->forward_project(*projdata2, *image2_sptr);
    
    int view1_num = 0, view2_num = 0;
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB1;
    for(int i=0;i<projdata->get_max_view_num();i++){
        Bin bin(0,i,0,0);
        proj_data_info_blocks_ptr->get_LOR(lorB1,bin);
        if(abs(lorB1.phi()-phi1)/phi1<=1E-2){
            view1_num=i;
            break;
        }
    }
    
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB2;
    for(int i=0;i<projdata2->get_max_view_num();i++){
        Bin bin(0,i,0,0);
        proj_data_info_blocks_ptr->get_LOR(lorB2,bin);
        if(abs(lorB2.phi()-phi2)/phi2<=1E-2){
            view2_num=i;
            break;
        }
    }
    
    float max1 =projdata->get_sinogram(0,0).find_max();
    float max2 =projdata2->get_sinogram(0,0).find_max();
    
//    find the tang position with the max value
    int tang1_num=0,tang2_num=0;
        for(int tang=projdata->get_min_tangential_pos_num();tang<projdata->get_max_tangential_pos_num();tang++){
            
            if((max1-projdata->get_sinogram(0,0).at(view1_num).at(tang))/max1<1E-3) {
                tang1_num=tang;
                break;
            }
        }

        for(int tang=projdata2->get_min_tangential_pos_num();tang<projdata2->get_max_tangential_pos_num();tang++){
            
            if((max2-projdata2->get_sinogram(0,0).at(view2_num).at(tang))/max2<1E-3) {
                tang2_num=tang;
                break;
            }
        }

        float bin1=projdata->get_sinogram(0,0).at(view1_num).at(tang1_num);
        float bin2=projdata2->get_sinogram(0,0).at(view2_num).at(tang2_num);
    set_tolerance(10E-2);
    check_if_equal(bin1, max1,"the value seen in the block at 60 degrees should be the same as the max value of the sinogram");
    check_if_equal(bin2, max2,"the value seen in the block at 30 degrees should be the same as the max value of the sinogram");
}

/*! The following is a test for symmetries: a simulated image is created with spherical source in front of each detector block,
 *  the forward projected sinogram should show the same bin values in simmetric places (in this test a dodecagone scanner is 
 * used so we have symmetry every 30 degrees. The above is repeated for an image with sources in front of the dodecagone corners. 
 * the sinogram should have now different values at fixed bin compared the previous image
*/
void
BlocksTests::run_symmetry_test(){
    
    CartesianCoordinate3D<float> origin (0,0,0);  
    CartesianCoordinate3D<float> grid_spacing (1.1,2.2,2.2); 
    float theta1=0;float theta2=0;
    
    const IndexRange<3> 
      range(Coordinate3D<int>(0,-45,-44),
            Coordinate3D<int>(24,44,45));
    VoxelsOnCartesianGrid<float>  image(range,origin, grid_spacing);
    
    const Array<2,float> direction_vectors=
  make_array(make_1d_array(1.F,0.F,0.F),
         make_1d_array(0.F,cos(theta1),sin(theta1)),
         make_1d_array(0.F,-sin(theta1),cos(theta1)));

    Ellipsoid
      ellipsoid(CartesianCoordinate3D<float>(/*radius_z*/6*grid_spacing.z(),
                                             /*radius_y*/6*grid_spacing.y(),
                                             /*radius_x*/6*grid_spacing.x()),
		        /*centre*/CartesianCoordinate3D<float>((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(),
                                                       -34*grid_spacing.y(),
                                                       0),
                direction_vectors);
    
    ellipsoid.construct_volume(image, make_coordinate(3,3,3));
    
    VoxelsOnCartesianGrid<float>  image1=image;
    VoxelsOnCartesianGrid<float>  image22=image;
//    rotate by 30 degrees, this scanner is a dodecagon and there is a 30 degrees angle between consecutive blocks
    for(int i=30; i<360; i+=30){
        theta1=i*_PI/180;
    
        CartesianCoordinate3D<float> origin1 ((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(),
                                              -34*grid_spacing.y()*cos(theta1),
                                              34*grid_spacing.y()*sin(theta1));
    
        ellipsoid.set_origin(origin1);
        ellipsoid.construct_volume(image1, make_coordinate(3,3,3));
        image+=image1;
}
    shared_ptr<DiscretisedDensity<3,float> > image1_sptr(image.clone());
    write_to_file("image_for",*image1_sptr);
    
    
    image=*image.get_empty_copy();
    for(int i=15; i<360; i+=30){
        theta2=i*_PI/180;
    
        CartesianCoordinate3D<float> origin2 ((image.get_min_index()+image.get_max_index())/2*grid_spacing.z(),
                                              -34*grid_spacing.y()*cos(theta2),
                                              34*grid_spacing.y()*sin(theta2));
    
        ellipsoid.set_origin(origin2);
        ellipsoid.construct_volume(image22, make_coordinate(3,3,3));
        image+=image22;
}

shared_ptr<DiscretisedDensity<3,float> > image2_sptr(image.clone());
write_to_file("image_for2",*image2_sptr);



//    create projadata info
    shared_ptr<Scanner> scannerBlocks_ptr;
    scannerBlocks_ptr.reset(new Scanner (Scanner::SAFIRDualRingPrototype));
    scannerBlocks_ptr->set_num_axial_crystals_per_block(1);
    scannerBlocks_ptr->set_axial_block_spacing(scannerBlocks_ptr->get_axial_crystal_spacing()*
                                               scannerBlocks_ptr->get_num_axial_crystals_per_block());
    scannerBlocks_ptr->set_transaxial_block_spacing(scannerBlocks_ptr->get_transaxial_crystal_spacing()*
                                                    scannerBlocks_ptr->get_num_transaxial_crystals_per_block());
//    scannerBlocks_ptr->set_num_transaxial_crystals_per_block(1);
    scannerBlocks_ptr->set_num_axial_blocks_per_bucket(1);
//    scannerBlocks_ptr->set_num_transaxial_blocks_per_bucket(1);
    scannerBlocks_ptr->set_num_rings(1);
    
    scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
    
    
    VectorWithOffset<int> num_axial_pos_per_segment(scannerBlocks_ptr->get_num_rings()*2-1);
    VectorWithOffset<int> min_ring_diff_v(scannerBlocks_ptr->get_num_rings()*2-1);
    VectorWithOffset<int> max_ring_diff_v(scannerBlocks_ptr->get_num_rings()*2-1);
    
    for (int i=0; i<2*scannerBlocks_ptr->get_num_rings()-1; i++){
        min_ring_diff_v[i]=-scannerBlocks_ptr->get_num_rings()+1+i;
        max_ring_diff_v[i]=-scannerBlocks_ptr->get_num_rings()+1+i;
        if (i<scannerBlocks_ptr->get_num_rings())
            num_axial_pos_per_segment[i]=i+1;
        else
            num_axial_pos_per_segment[i]=2*scannerBlocks_ptr->get_num_rings()-i-1;
        }
    
    shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_blocks_ptr;
    proj_data_info_blocks_ptr.reset(
                new ProjDataInfoBlocksOnCylindricalNoArcCorr(
                    scannerBlocks_ptr,
                    num_axial_pos_per_segment,
                    min_ring_diff_v, max_ring_diff_v,
                    scannerBlocks_ptr->get_max_num_views(),
                    scannerBlocks_ptr->get_max_num_non_arccorrected_bins()));

//    now forward-project images
    
    shared_ptr<ForwardProjectorByBin> forw_projector1_sptr, forw_projector2_sptr;
    shared_ptr<ProjData> projdata1, projdata2;
    shared_ptr<ProjMatrixByBin> PM(new  ProjMatrixByBinUsingRayTracing());
    forw_projector1_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    info(boost::format("Test blocks on Cylindrical: Forward projector used: %1%") % forw_projector1_sptr->parameter_info());
    
    forw_projector1_sptr->set_up(proj_data_info_blocks_ptr,
                                image1_sptr);
    
    
    forw_projector2_sptr.reset(new ForwardProjectorByBinUsingProjMatrixByBin(PM));
    forw_projector2_sptr->set_up(proj_data_info_blocks_ptr,
                                image2_sptr);

    //-- ExamInfo
    shared_ptr<ExamInfo> exam_info_sptr(new ExamInfo());
    exam_info_sptr->imaging_modality = ImagingModality::PT;
    
    projdata1.reset(new ProjDataInterfile(exam_info_sptr,
                                             proj_data_info_blocks_ptr,
                                         "sino1_from_image.hs",std::ios::out | std::ios::trunc | std::ios::in));
    
    projdata2.reset(new ProjDataInterfile(exam_info_sptr,
                                             proj_data_info_blocks_ptr,
                                         "sino2_from_image.hs",std::ios::out | std::ios::trunc | std::ios::in));
    
    forw_projector1_sptr->forward_project(*projdata1, *image1_sptr);
    forw_projector2_sptr->forward_project(*projdata2, *image2_sptr);
    int crystals_in_ring=scannerBlocks_ptr->get_num_detectors_per_ring();
    float bin1_0=projdata1->get_sinogram(0,0).at(0/crystals_in_ring*_PI).at(0);
    float bin1_90=projdata1->get_sinogram(0,0).at(90/crystals_in_ring*_PI).at(0);
    float bin1_30=projdata1->get_sinogram(0,0).at(30/crystals_in_ring*_PI).at(0);
    float bin1_60=projdata1->get_sinogram(0,0).at(60/crystals_in_ring*_PI).at(0);
    float bin1_150=projdata1->get_sinogram(0,0).at(150/crystals_in_ring*_PI).at(0);
     
//    values of the asymetric image
    float bin2_0=projdata2->get_sinogram(0,0).at(0/crystals_in_ring*_PI).at(0);
    float bin2_90=projdata2->get_sinogram(0,0).at(90/crystals_in_ring*_PI).at(0);
    float bin2_30=projdata2->get_sinogram(0,0).at(30/crystals_in_ring*_PI).at(0);
    float bin2_60=projdata2->get_sinogram(0,0).at(60/crystals_in_ring*_PI).at(0);
    float bin2_150=projdata2->get_sinogram(0,0).at(150/crystals_in_ring*_PI).at(0);
    
    set_tolerance(10E-3);
    check_if_equal(bin1_0, bin1_90,"the value seen in the block 0 should be the same as the one at angle 90");
    check_if_equal(bin1_30, bin1_150,"the value seen in the block at angle 30 should be the same as the one at angle 150 ");
    check_if_equal(bin1_30, bin1_60,"the value seen in the block at angle 30 should be the same as the one at angle 60");
    

    check(bin1_0!= bin2_0,"the two data have different symmetries, the values should be different");
    check(bin1_30!= bin2_30,"the two data have different symmetries, the values should be different ");
    check(bin1_60!= bin2_60,"the two data have different symmetries, the values should be different");
    check(bin1_90!= bin2_90,"the two data have different symmetries, the values should be different");
    check(bin1_30!= bin2_30,"the two data have different symmetries, the values should be different");
    check(bin1_150!= bin2_150,"the two data have different symmetries, the values should be different");
}

void
BlocksTests::
run_tests()
{
    
    std::cerr << "-------- Testing Blocks Geometry --------\n";
    run_symmetry_test();
    run_plane_symmetry_test();
}
END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  BlocksTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
