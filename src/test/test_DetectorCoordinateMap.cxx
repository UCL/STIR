/*!

  \file
  \ingroup recontest

  \brief Test program for detection position map using stir::ProjDataInfoBlockOnCylindrical

  \author Daniel Deidda

*/
/*  Copyright (C) 2021, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/

#include "stir/info.h"
#include "stir/make_array.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ExamInfo.h"
#include "stir/DetectorCoordinateMap.h"
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
class DetectionPosMapTests: public RunTests
{
public:
  void run_tests();
  float calculate_angle_within_half_bucket(const shared_ptr<Scanner> scanner_ptr,
                                           const shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr);
private:
  void run_coordinate_test_for_flat_first_bucket();
  void run_map_orientation_test();
};

float
DetectionPosMapTests::calculate_angle_within_half_bucket(const shared_ptr<Scanner> scanner_ptr,
                                                         const shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_ptr){
    Bin bin;
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB;
    float csi;
    float C_spacing=scanner_ptr->get_transaxial_crystal_spacing();
    float csi_crystal=std::atan((C_spacing)/scanner_ptr->get_effective_ring_radius());
//    float bucket_spacing=scanner_ptr->get_transaxial_block_spacing()*C_spacing;
//    float blocks_gap=scanner_ptr->get_transaxial_block_spacing()
//            -scanner_ptr->get_num_transaxial_crystals_per_block()*C_spacing;
//    float csi_gap=std::atan((blocks_gap)/scanner_ptr->get_effective_ring_radius());
    
//    get angle within half bucket
    for (int view = 0; view <= scanner_ptr->get_max_num_views(); view++){
        int bucket_num=view/(scanner_ptr->get_num_transaxial_crystals_per_block()*scanner_ptr->get_num_transaxial_blocks_per_bucket());
        if (bucket_num>0)
            break;
        
        bin.segment_num() = 0;
        bin.axial_pos_num() = 0;
        bin.view_num() = view;
        bin.tangential_pos_num() = 0;
        
        proj_data_info_ptr->get_LOR(lorB,bin);
        csi=lorB.phi();
    }
    return (csi+csi_crystal)/2;
}

/*!
  The following test checks that the y position of the detectors in buckets that are parallel to the x axis are the same. 
  The calculation of csi is only valid for the scanner defined in stir::Scanner, if we modify the number of blocks per bucket
  csi will be affected. However this does not happen when csi is calculated in the same way we do in the crystal map.
*/
void
DetectionPosMapTests::run_coordinate_test_for_flat_first_bucket()
{
    CPUTimer timer;
    shared_ptr<Scanner> scannerBlocks_ptr;
    scannerBlocks_ptr.reset(new Scanner (Scanner::SAFIRDualRingPrototype));
    
    scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
    scannerBlocks_ptr->set_num_transaxial_blocks_per_bucket(1);
    scannerBlocks_ptr->set_up();

    

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

    Bin bin, bin0=Bin(0,0,0,0);
    CartesianCoordinate3D< float> b1,b2,b01,b02;
    
    //    estimate the angle covered by half bucket, csi
    float csi;
    csi=calculate_angle_within_half_bucket(scannerBlocks_ptr,
                                           proj_data_info_blocks_ptr);
    
    shared_ptr<Scanner> scannerBlocks_firstFlat_ptr;
    scannerBlocks_firstFlat_ptr.reset(new Scanner (Scanner::SAFIRDualRingPrototype));
    scannerBlocks_firstFlat_ptr->set_scanner_geometry("BlocksOnCylindrical");
    scannerBlocks_firstFlat_ptr->set_num_transaxial_blocks_per_bucket(1);
    scannerBlocks_firstFlat_ptr->set_intrinsic_azimuthal_tilt(-csi);
    scannerBlocks_firstFlat_ptr->set_up();
    
    shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_blocks_firstFlat_ptr;
    proj_data_info_blocks_firstFlat_ptr.reset(
                new ProjDataInfoBlocksOnCylindricalNoArcCorr(
                    scannerBlocks_firstFlat_ptr,
                    num_axial_pos_per_segment,
                    min_ring_diff_v, max_ring_diff_v,
                    scannerBlocks_firstFlat_ptr->get_max_num_views(),
                    scannerBlocks_firstFlat_ptr->get_max_num_non_arccorrected_bins()));
    timer.reset(); timer.start();
    
    for (int view = 0; view <= proj_data_info_blocks_firstFlat_ptr->get_max_view_num(); view++)
    {
        int bucket_num=view/(scannerBlocks_firstFlat_ptr->get_num_transaxial_crystals_per_block()*
                             scannerBlocks_firstFlat_ptr->get_num_transaxial_blocks_per_bucket());
        if (bucket_num>0)
            break;
        
        bin.segment_num() = 0;
        bin.axial_pos_num() = 0;
        bin.view_num() = view;
        bin.tangential_pos_num() = 0;
        
        
        //                check cartesian coordinates of detectors
        proj_data_info_blocks_firstFlat_ptr->find_cartesian_coordinates_of_detection(b1,b2,bin);
        proj_data_info_blocks_firstFlat_ptr->find_cartesian_coordinates_of_detection(b01,b02,bin0);
        
        
        check_if_equal(b1.y(),b01.y(), " checking cartesian coordinate y1 are the same on a flat bucket");
        check_if_equal(b2.y(),b02.y(), " checking cartesian coordinate y2 are the same on a flat bucket");
        check_if_equal(b1.y(),-b2.y(), " checking cartesian coordinate y1 and y2 are of opposite sign on opposite flat buckets");
        
    }
    timer.stop(); std::cerr<< "-- CPU Time " << timer.value() << '\n';
    
}

void
DetectionPosMapTests::run_map_orientation_test()
{
    CPUTimer timer;
    shared_ptr<const DetectorCoordinateMap> map_ptr;
    shared_ptr<Scanner> scannerBlocks_ptr;
    scannerBlocks_ptr.reset(new Scanner (Scanner::SAFIRDualRingPrototype));
    
    scannerBlocks_ptr->set_scanner_geometry("BlocksOnCylindrical");
    scannerBlocks_ptr->set_num_transaxial_blocks_per_bucket(1);
    scannerBlocks_ptr->set_up();

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

    Bin bin, bin0=Bin(0,0,0,0);
    CartesianCoordinate3D< float> b1,b2,rb1,rb2;
    DetectionPosition<> det_pos, det_pos_ord;
    CartesianCoordinate3D<float> coord_ord;
    map_ptr=scannerBlocks_ptr->get_detector_map_sptr();
    int rad_size=map_ptr->get_num_radial_coords();
    int ax_size=map_ptr->get_num_axial_coords();
    int tang_size=map_ptr->get_num_tangential_coords();
    
    DetectorCoordinateMap::det_pos_to_coord_type coord_map_reordered;
    
//    reorder the tangential positions
    for (int rad=0;rad<rad_size; rad++)
        for (int ax=0;ax<ax_size; ax++)
            for (int tang=0;tang<tang_size; tang++)
            {
                det_pos.radial_coord()=rad;
                det_pos.axial_coord()=ax;
                det_pos.tangential_coord()=tang;
                det_pos_ord.radial_coord()=rad;
                det_pos_ord.axial_coord()=ax;
                det_pos_ord.tangential_coord()=tang_size-1-tang;
                
                coord_ord=map_ptr->get_coordinate_for_det_pos(det_pos_ord);
                coord_map_reordered[det_pos]=coord_ord;
            }
    
    shared_ptr<Scanner> scannerBlocks_reord_ptr;
    scannerBlocks_reord_ptr.reset(new Scanner (Scanner::SAFIRDualRingPrototype));
    scannerBlocks_reord_ptr->set_scanner_geometry("BlocksOnCylindrical");
    scannerBlocks_reord_ptr->set_num_transaxial_blocks_per_bucket(1);
    scannerBlocks_reord_ptr->set_detector_map(coord_map_reordered);
    scannerBlocks_reord_ptr->set_up();
    
    shared_ptr<ProjDataInfoBlocksOnCylindricalNoArcCorr> proj_data_info_blocks_reord_ptr;
    proj_data_info_blocks_reord_ptr.reset(
                new ProjDataInfoBlocksOnCylindricalNoArcCorr(
                    scannerBlocks_reord_ptr,
                    num_axial_pos_per_segment,
                    min_ring_diff_v, max_ring_diff_v,
                    scannerBlocks_reord_ptr->get_max_num_views(),
                    scannerBlocks_reord_ptr->get_max_num_non_arccorrected_bins()));
    timer.reset(); timer.start();
    
    for (int view = 0; view <= proj_data_info_blocks_reord_ptr->get_max_view_num(); view++)
    {
                
        bin.segment_num() = 0;
        bin.axial_pos_num() = 0;
        bin.view_num() = view;
        bin.tangential_pos_num() = 0;
        
//        bin_ord.segment_num() = 0;
//        bin_ord.axial_pos_num() = 0;
//        bin_ord.view_num() = proj_data_info_blocks_reord_ptr->get_max_view_num()-view;
//        bin_ord.tangential_pos_num() = 0;
        
        
//        //                check cartesian coordinates of detectors
        proj_data_info_blocks_ptr->find_cartesian_coordinates_of_detection(b1,b2,bin);
        proj_data_info_blocks_reord_ptr->find_cartesian_coordinates_of_detection(rb1,b2,bin);
        
        
        check_if_equal(b1,rb1, " checking cartesian coordinate y1 are the same on a flat bucket");
    }
    timer.stop(); std::cerr<< "-- CPU Time " << timer.value() << '\n';
    
}


void
DetectionPosMapTests::
run_tests()
{
    
    std::cerr << "-------- Testing DetectorCoordinateMap --------\n";
    run_map_orientation_test();
    run_coordinate_test_for_flat_first_bucket();
}
END_NAMESPACE_STIR


USING_NAMESPACE_STIR

int main()
{
  DetectionPosMapTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
