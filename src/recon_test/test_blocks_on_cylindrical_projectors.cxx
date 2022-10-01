/*!

  \file
  \ingroup recontest

  \brief Test program for back projection and forward projection using stir::ProjDataInfoBlockOnCylindrical

  \author Daniel Deidda

*/
/*  Copyright (C) 2021-2022, National Physical Laboratory
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
#include "stir/make_array.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ExamInfo.h"
#include "stir/Verbosity.h"
#include "stir/LORCoordinates.h"
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
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
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/IO/write_to_file.h"
#include "stir/VoxelsOnCartesianGrid.h"
//#include "stir/Shape/Shape3D.h"
#include <cmath>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief Test class for Blocks
*/
class BlocksTests : public RunTests
{
public:
  void run_tests();

private:
  void run_symmetry_test();
  void run_plane_symmetry_test();
  void run_map_orientation_test();
  void run_axial_projection_test();
  void run_voxelOnCartesianGrid_with_negative_offset();
  void run_intersection_with_cylinder_test();
  template <class TProjDataInfo>
  shared_ptr<TProjDataInfo> set_blocks_projdata_info(shared_ptr<Scanner> scanner_sptr);
};
/*! The following is a function to allow a projdata_info blocksONCylindrical to be created from the scanner.
 */
template <class TProjDataInfo>
shared_ptr<TProjDataInfo>
BlocksTests::set_blocks_projdata_info(shared_ptr<Scanner> scanner_sptr)
{
  auto segments = scanner_sptr->get_num_rings() - 1;
  VectorWithOffset<int> num_axial_pos_per_segment(-segments, segments);
  VectorWithOffset<int> min_ring_diff_v(-segments, segments);
  VectorWithOffset<int> max_ring_diff_v(-segments, segments);
  for (int i = -segments; i <= segments; i++)
    {
      min_ring_diff_v[i] = i;
      max_ring_diff_v[i] = i;
      num_axial_pos_per_segment[i] = scanner_sptr->get_num_rings() - abs(i);
    }

  auto proj_data_info_blocks_sptr
      = std::make_shared<TProjDataInfo>(scanner_sptr, num_axial_pos_per_segment, min_ring_diff_v, max_ring_diff_v,
                                        scanner_sptr->get_max_num_views(), scanner_sptr->get_max_num_non_arccorrected_bins());

  return proj_data_info_blocks_sptr;
}

/*! The following is a test for the view offset with voxelOnCartesianGrid: ascanner is created and its relative
 * projdata info using a negative offset is set. When calling the voxelOnCartesianGrid constructor if everything ok
 * a variable called is_OK is set to true and false otherwise. If false the test will fail.
 */
void
BlocksTests::run_voxelOnCartesianGrid_with_negative_offset()
{
  //    create projadata info

  auto scannerBlocks_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);

  scannerBlocks_sptr->set_intrinsic_azimuthal_tilt(-30);
  scannerBlocks_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_sptr->set_up();

  auto proj_data_info_blocks_sptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_sptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_sptr);

  bool is_ok = true;

  try
    {
      auto grid = std::make_shared<VoxelsOnCartesianGrid<float>>(
          *proj_data_info_blocks_sptr, 1, CartesianCoordinate3D<float>(0.F, 0.F, 0.F), CartesianCoordinate3D<int>(-1, -1, -1));
    }
  catch (...)
    {
      is_ok = false;
    }

  check_if_equal(is_ok, true);
}

/*! The following is a test for axial symmetries: a simulated image is created with a line along the z direction.
 *  The image is forward projected to a sinogram and the sinogram back projected to an image. This image should
 *  be symmetrical along z
 */
void
BlocksTests::run_axial_projection_test()
{

  //-- ExamInfo
  auto exam_info_sptr = std::make_shared<ExamInfo>();
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  CartesianCoordinate3D<float> origin(0, 0, 0);
  CartesianCoordinate3D<float> grid_spacing(1.1, 2.2, 2.2);

  const IndexRange<3> range(Coordinate3D<int>(0, -45, -45), Coordinate3D<int>(24, 44, 44));
  VoxelsOnCartesianGrid<float> image(exam_info_sptr, range, origin, grid_spacing);

  //    60 degrees
  float phi1 = 0 * _PI / 180;
  const Array<2, float> direction_vectors
      = make_array(make_1d_array(1.F, 0.F, 0.F), make_1d_array(0.F, cos(float(_PI) - phi1), sin(float(_PI) - phi1)),
                   make_1d_array(0.F, -sin(float(_PI) - phi1), cos(float(_PI) - phi1)));

  Ellipsoid plane(CartesianCoordinate3D<float>(/*edge_z*/ 50 * grid_spacing.z(),
                                               /*edge_y*/ 2 * grid_spacing.y(),
                                               /*edge_x*/ 2 * grid_spacing.x()),
                  /*centre*/
                  CartesianCoordinate3D<float>((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                               0 * grid_spacing.y(), 0),
                  direction_vectors);

  plane.construct_volume(image, make_coordinate(3, 3, 3));

  //    create projadata info

  auto scannerBlocks_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_sptr->set_up();

  auto proj_data_info_blocks_sptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_sptr
      = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_sptr); //    now forward-project image

  shared_ptr<DiscretisedDensity<3, float>> image_sptr(image.clone());
  shared_ptr<DiscretisedDensity<3, float>> bck_proj_image_sptr(image.clone());
  write_to_file("axial_test", *image_sptr);

  auto PM = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  PM->enable_cache(false);
  //    PM->set_do_symmetry_90degrees_min_phi(false);
  //    PM->set_do_symmetry_shift_z(false);
  //    PM->set_do_symmetry_swap_segment(false);

  auto forw_projector_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  auto bck_projector_sptr = std::make_shared<BackProjectorByBinUsingProjMatrixByBin>(PM);
  info(boost::format("Test blocks on Cylindrical: Forward projector used: %1%") % forw_projector_sptr->parameter_info());

  forw_projector_sptr->set_up(proj_data_info_blocks_sptr, image_sptr);
  bck_projector_sptr->set_up(proj_data_info_blocks_sptr, bck_proj_image_sptr);

  auto projdata = std::make_shared<ProjDataInterfile>(exam_info_sptr, proj_data_info_blocks_sptr, "test_axial.hs",
                                                      std::ios::out | std::ios::trunc | std::ios::in);

  forw_projector_sptr->forward_project(*projdata, *image_sptr);

  bck_projector_sptr->back_project(*bck_proj_image_sptr, *projdata, 0, 1);
  write_to_file("back_proj_axial_test", *bck_proj_image_sptr);

  int min_z = bck_proj_image_sptr->get_min_index();
  int max_z = bck_proj_image_sptr->get_max_index();
  int min_y = (*bck_proj_image_sptr)[min_z].get_min_index();
  int max_y = (*bck_proj_image_sptr)[min_z].get_max_index();
  int min_x = (*bck_proj_image_sptr)[min_z][min_y].get_min_index();
  int max_x = (*bck_proj_image_sptr)[min_z][min_y].get_max_index();

  // get two planes in the image that are equidistant from the z center
  int centre_z = (max_z - min_z) / 2;
  int plane_idA = centre_z - 5;
  int plane_idB = centre_z + 5;

  for (int y = min_y; y < max_y; y++)
    for (int x = min_x; x < max_x; x++)
      {
        check_if_equal((*bck_proj_image_sptr)[plane_idA][y][x], (*bck_proj_image_sptr)[plane_idB][y][x],
                       "checking the symmetry along the axial direction");
      }
}

/*! The following is a test for symmetries: a simulated image is created with a plane at known angles,
 *  the forward projected sinogram should show the maximum value at the bin corresponding to the angle phi
 *  equal to the orientation of the plane
 */
void
BlocksTests::run_plane_symmetry_test()
{

  //-- ExamInfo
  auto exam_info_sptr = std::make_shared<ExamInfo>();
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  CartesianCoordinate3D<float> origin(0, 0, 0);
  CartesianCoordinate3D<float> grid_spacing(1.1, 2.2, 2.2);
  float phi1;
  float phi2;
  const IndexRange<3> range(Coordinate3D<int>(0, -45, -44), Coordinate3D<int>(24, 44, 45));
  VoxelsOnCartesianGrid<float> image(exam_info_sptr, range, origin, grid_spacing);

  //    60 degrees
  phi1 = 60 * _PI / 180;
  const Array<2, float> direction_vectors
      = make_array(make_1d_array(1.F, 0.F, 0.F), make_1d_array(0.F, cos(float(_PI) - phi1), sin(float(_PI) - phi1)),
                   make_1d_array(0.F, -sin(float(_PI) - phi1), cos(float(_PI) - phi1)));

  Ellipsoid plane(CartesianCoordinate3D<float>(/*edge_z*/ 25 * grid_spacing.z(),
                                               /*edge_y*/ 91 * grid_spacing.y(),
                                               /*edge_x*/ 5 * grid_spacing.x()),
                  /*centre*/
                  CartesianCoordinate3D<float>((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                               0 * grid_spacing.y(), 0),
                  direction_vectors);

  plane.construct_volume(image, make_coordinate(3, 3, 3));

  //    rotate by 30 degrees
  phi2 = 30 * _PI / 180;
  VoxelsOnCartesianGrid<float> image2 = *image.get_empty_copy();
  const Array<2, float> direction2
      = make_array(make_1d_array(1.F, 0.F, 0.F), make_1d_array(0.F, cos(float(_PI) - phi2), sin(float(_PI) - phi2)),
                   make_1d_array(0.F, -sin(float(_PI) - phi2), cos(float(_PI) - phi2)));

  Ellipsoid plane2(CartesianCoordinate3D<float>(/*edge_z*/ 25 * grid_spacing.z(),
                                                /*edge_y*/ 91 * grid_spacing.y(),
                                                /*edge_x*/ 5 * grid_spacing.x()),
                   /*centre*/
                   CartesianCoordinate3D<float>((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                                0 * grid_spacing.y(), 0),
                   direction2);
  //    plane.set_direction_vectors(direction2);

  plane2.construct_volume(image2, make_coordinate(3, 3, 3));

  //    create projadata info

  auto scannerBlocks_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_sptr->set_num_axial_crystals_per_block(1);
  scannerBlocks_sptr->set_axial_block_spacing(scannerBlocks_sptr->get_axial_crystal_spacing()
                                              * scannerBlocks_sptr->get_num_axial_crystals_per_block());
  scannerBlocks_sptr->set_transaxial_block_spacing(scannerBlocks_sptr->get_transaxial_crystal_spacing()
                                                   * scannerBlocks_sptr->get_num_transaxial_crystals_per_block());
  //    scannerBlocks_sptr->set_num_transaxial_crystals_per_block(1);
  scannerBlocks_sptr->set_num_axial_blocks_per_bucket(2);
  //    scannerBlocks_sptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_sptr->set_num_rings(2);

  scannerBlocks_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_sptr->set_up();

  auto proj_data_info_blocks_sptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_sptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_sptr);

  //    now forward-project image

  shared_ptr<DiscretisedDensity<3, float>> image_sptr(image.clone());
  write_to_file("plane60", *image_sptr);

  shared_ptr<DiscretisedDensity<3, float>> image2_sptr(image2.clone());
  write_to_file("plane30", *image2_sptr);

  auto PM = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  PM->enable_cache(false);
  auto forw_projector_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  info(boost::format("Test blocks on Cylindrical: Forward projector used: %1%") % forw_projector_sptr->parameter_info());

  forw_projector_sptr->set_up(proj_data_info_blocks_sptr, image_sptr);

  auto forw_projector2_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  forw_projector2_sptr->set_up(proj_data_info_blocks_sptr, image2_sptr);

  auto projdata = std::make_shared<ProjDataInterfile>(exam_info_sptr, proj_data_info_blocks_sptr, "sino1_from_plane.hs",
                                                      std::ios::out | std::ios::trunc | std::ios::in);

  forw_projector_sptr->forward_project(*projdata, *image_sptr);

  auto projdata2 = std::make_shared<ProjDataInterfile>(exam_info_sptr, proj_data_info_blocks_sptr, "sino2_from_plane.hs",
                                                       std::ios::out | std::ios::trunc | std::ios::in);

  forw_projector2_sptr->forward_project(*projdata2, *image2_sptr);

  int view1_num = 0, view2_num = 0;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB1;
  for (int i = 0; i < projdata->get_max_view_num(); i++)
    {
      Bin bin(0, i, 0, 0);
      proj_data_info_blocks_sptr->get_LOR(lorB1, bin);
      if (abs(lorB1.phi() - phi1) / phi1 <= 1E-2)
        {
          view1_num = i;
          break;
        }
    }

  LORInAxialAndNoArcCorrSinogramCoordinates<float> lorB2;
  for (int i = 0; i < projdata2->get_max_view_num(); i++)
    {
      Bin bin(0, i, 0, 0);
      proj_data_info_blocks_sptr->get_LOR(lorB2, bin);
      if (abs(lorB2.phi() - phi2) / phi2 <= 1E-2)
        {
          view2_num = i;
          break;
        }
    }

  float max1 = projdata->get_sinogram(0, 0).find_max();
  float max2 = projdata2->get_sinogram(0, 0).find_max();

  //    find the tang position with the max value
  int tang1_num = 0, tang2_num = 0;
  for (int tang = projdata->get_min_tangential_pos_num(); tang < projdata->get_max_tangential_pos_num(); tang++)
    {

      if ((max1 - projdata->get_sinogram(0, 0).at(view1_num).at(tang)) / max1 < 1E-3)
        {
          tang1_num = tang;
          break;
        }
    }

  for (int tang = projdata2->get_min_tangential_pos_num(); tang < projdata2->get_max_tangential_pos_num(); tang++)
    {

      if ((max2 - projdata2->get_sinogram(0, 0).at(view2_num).at(tang)) / max2 < 1E-3)
        {
          tang2_num = tang;
          break;
        }
    }

  float bin1 = projdata->get_sinogram(0, 0).at(view1_num).at(tang1_num);
  float bin2 = projdata2->get_sinogram(0, 0).at(view2_num).at(tang2_num);
  set_tolerance(10E-2);
  check_if_equal(bin1, max1, "the value seen in the block at 60 degrees should be the same as the max value of the sinogram");
  check_if_equal(bin2, max2, "the value seen in the block at 30 degrees should be the same as the max value of the sinogram");
}

/*! The following is a test for symmetries: a simulated image is created with spherical source in front of each detector block,
 *  the forward projected sinogram should show the same bin values in symmetric places (in this test a dodecagon scanner is
 * used so we have symmetry every 30 degrees. The above is repeated for an image with sources in front of the dodecagon corners.
 * The sinogram should have now different values at fixed bin compared the previous image. In this test we are also testing a
 * projdata_info with a negative view offset.
 */
void
BlocksTests::run_symmetry_test()
{

  //-- ExamInfo
  auto exam_info_sptr = std::make_shared<ExamInfo>();
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  CartesianCoordinate3D<float> origin(0, 0, 0);
  CartesianCoordinate3D<float> grid_spacing(1.1, 2.2, 2.2);
  float theta1 = 0;
  float theta2 = 0;

  const IndexRange<3> range(Coordinate3D<int>(0, -45, -44), Coordinate3D<int>(24, 44, 45));
  VoxelsOnCartesianGrid<float> image(exam_info_sptr, range, origin, grid_spacing);

  const Array<2, float> direction_vectors = make_array(make_1d_array(1.F, 0.F, 0.F), make_1d_array(0.F, cos(theta1), sin(theta1)),
                                                       make_1d_array(0.F, -sin(theta1), cos(theta1)));

  Ellipsoid ellipsoid(CartesianCoordinate3D<float>(/*radius_z*/ 6 * grid_spacing.z(),
                                                   /*radius_y*/ 6 * grid_spacing.y(),
                                                   /*radius_x*/ 6 * grid_spacing.x()),
                      /*centre*/
                      CartesianCoordinate3D<float>((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                                   -34 * grid_spacing.y(), 0),
                      direction_vectors);

  ellipsoid.construct_volume(image, make_coordinate(3, 3, 3));

  VoxelsOnCartesianGrid<float> image1 = image;
  VoxelsOnCartesianGrid<float> image22 = image;
  //    rotate by 30 degrees, this scanner is a dodecagon and there is a 30 degrees angle between consecutive blocks
  for (int i = 30; i < 360; i += 30)
    {
      theta1 = i * _PI / 180;

      CartesianCoordinate3D<float> origin1((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                           -34 * grid_spacing.y() * cos(theta1), 34 * grid_spacing.y() * sin(theta1));

      ellipsoid.set_origin(origin1);
      ellipsoid.construct_volume(image1, make_coordinate(3, 3, 3));
      image += image1;
    }
  shared_ptr<DiscretisedDensity<3, float>> image1_sptr(image.clone());
  write_to_file("image_for", *image1_sptr);

  image = *image.get_empty_copy();
  for (int i = 15; i < 360; i += 30)
    {
      theta2 = i * _PI / 180;

      CartesianCoordinate3D<float> origin2((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                           -34 * grid_spacing.y() * cos(theta2), 34 * grid_spacing.y() * sin(theta2));

      ellipsoid.set_origin(origin2);
      ellipsoid.construct_volume(image22, make_coordinate(3, 3, 3));
      image += image22;
    }

  shared_ptr<DiscretisedDensity<3, float>> image2_sptr(image.clone());
  write_to_file("image_for2", *image2_sptr);

  //    create projadata info
  auto scannerBlocks_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_sptr->set_num_axial_crystals_per_block(1);
  scannerBlocks_sptr->set_axial_block_spacing(scannerBlocks_sptr->get_axial_crystal_spacing()
                                              * scannerBlocks_sptr->get_num_axial_crystals_per_block());
  scannerBlocks_sptr->set_transaxial_block_spacing(scannerBlocks_sptr->get_transaxial_crystal_spacing()
                                                   * scannerBlocks_sptr->get_num_transaxial_crystals_per_block());
  //    scannerBlocks_sptr->set_num_transaxial_crystals_per_block(1);
  scannerBlocks_sptr->set_num_axial_blocks_per_bucket(2);
  //    scannerBlocks_sptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_sptr->set_num_rings(2);
  scannerBlocks_sptr->set_intrinsic_azimuthal_tilt(-30);
  scannerBlocks_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_sptr->set_up();

  auto proj_data_info_blocks_sptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_sptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_sptr);
  //    now forward-project images

  auto PM = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  PM->enable_cache(false);
  auto forw_projector1_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  info(boost::format("Test blocks on Cylindrical: Forward projector used: %1%") % forw_projector1_sptr->parameter_info());

  forw_projector1_sptr->set_up(proj_data_info_blocks_sptr, image1_sptr);

  auto forw_projector2_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  forw_projector2_sptr->set_up(proj_data_info_blocks_sptr, image2_sptr);

  auto projdata1 = std::make_shared<ProjDataInterfile>(exam_info_sptr, proj_data_info_blocks_sptr, "sino1_from_image.hs",
                                                       std::ios::out | std::ios::trunc | std::ios::in);

  auto projdata2 = std::make_shared<ProjDataInterfile>(exam_info_sptr, proj_data_info_blocks_sptr, "sino2_from_image.hs",
                                                       std::ios::out | std::ios::trunc | std::ios::in);

  forw_projector1_sptr->forward_project(*projdata1, *image1_sptr);
  forw_projector2_sptr->forward_project(*projdata2, *image2_sptr);
  int crystals_in_ring = scannerBlocks_sptr->get_num_detectors_per_ring();
  float bin1_0 = projdata1->get_sinogram(0, 0).at(0 / crystals_in_ring * _PI).at(0);
  float bin1_90 = projdata1->get_sinogram(0, 0).at(90 / crystals_in_ring * _PI).at(0);
  float bin1_30 = projdata1->get_sinogram(0, 0).at(30 / crystals_in_ring * _PI).at(0);
  float bin1_60 = projdata1->get_sinogram(0, 0).at(60 / crystals_in_ring * _PI).at(0);
  float bin1_150 = projdata1->get_sinogram(0, 0).at(150 / crystals_in_ring * _PI).at(0);

  //    values of the asymetric image
  float bin2_0 = projdata2->get_sinogram(0, 0).at(0 / crystals_in_ring * _PI).at(0);
  float bin2_90 = projdata2->get_sinogram(0, 0).at(90 / crystals_in_ring * _PI).at(0);
  float bin2_30 = projdata2->get_sinogram(0, 0).at(30 / crystals_in_ring * _PI).at(0);
  float bin2_60 = projdata2->get_sinogram(0, 0).at(60 / crystals_in_ring * _PI).at(0);
  float bin2_150 = projdata2->get_sinogram(0, 0).at(150 / crystals_in_ring * _PI).at(0);

  set_tolerance(10E-3);
  check_if_equal(bin1_0, bin1_90, "the value seen in the block 0 should be the same as the one at angle 90");
  check_if_equal(bin1_30, bin1_150, "the value seen in the block at angle 30 should be the same as the one at angle 150 ");
  check_if_equal(bin1_30, bin1_60, "the value seen in the block at angle 30 should be the same as the one at angle 60");

  check(bin1_0 != bin2_0, "the two data have different symmetries, the values should be different");
  check(bin1_30 != bin2_30, "the two data have different symmetries, the values should be different ");
  check(bin1_60 != bin2_60, "the two data have different symmetries, the values should be different");
  check(bin1_90 != bin2_90, "the two data have different symmetries, the values should be different");
  check(bin1_30 != bin2_30, "the two data have different symmetries, the values should be different");
  check(bin1_150 != bin2_150, "the two data have different symmetries, the values should be different");
}

/*!The following is a test for the crystal maps. Two scanners and ProjDataInfo are created, one with the standard map orientation
 * and the other with an orientation along the view which is opposite to the first one.  A simulated sphere was forward projected
 * to look at bin values in the two cases. The bin obtained from the two different projdata will have different coordinates but
 * the same value.
 */

void
BlocksTests::run_map_orientation_test()
{
  CPUTimer timer;
  //-- ExamInfo
  auto exam_info_sptr = std::make_shared<ExamInfo>();
  exam_info_sptr->imaging_modality = ImagingModality::PT;

  CartesianCoordinate3D<float> origin(0, 0, 0);
  CartesianCoordinate3D<float> grid_spacing(1.1, 2.2, 2.2);
  float theta1 = 0;

  const IndexRange<3> range(Coordinate3D<int>(0, -45, -44), Coordinate3D<int>(24, 44, 45));
  VoxelsOnCartesianGrid<float> image(exam_info_sptr, range, origin, grid_spacing);

  const Array<2, float> direction_vectors = make_array(make_1d_array(1.F, 0.F, 0.F), make_1d_array(0.F, cos(theta1), sin(theta1)),
                                                       make_1d_array(0.F, -sin(theta1), cos(theta1)));

  Ellipsoid ellipsoid(CartesianCoordinate3D<float>(/*radius_z*/ 6 * grid_spacing.z(),
                                                   /*radius_y*/ 6 * grid_spacing.y(),
                                                   /*radius_x*/ 6 * grid_spacing.x()),
                      /*centre*/
                      CartesianCoordinate3D<float>((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                                   -34 * grid_spacing.y(), 0),
                      direction_vectors);

  ellipsoid.construct_volume(image, make_coordinate(3, 3, 3));

  VoxelsOnCartesianGrid<float> image1 = image;
  VoxelsOnCartesianGrid<float> image22 = image;
  //    rotate by 30 degrees
  for (int i = 30; i < 90; i += 30)
    {
      theta1 = i * _PI / 180;

      CartesianCoordinate3D<float> origin1((image.get_min_index() + image.get_max_index()) / 2 * grid_spacing.z(),
                                           -34 * grid_spacing.y() * cos(theta1), 34 * grid_spacing.y() * sin(theta1));

      ellipsoid.set_origin(origin1);
      ellipsoid.construct_volume(image1, make_coordinate(3, 3, 3));
      image += image1;
    }
  shared_ptr<DiscretisedDensity<3, float>> image1_sptr(image.clone());
  write_to_file("image_to_fwp", *image1_sptr);

  image = *image.get_empty_copy();

  shared_ptr<const DetectorCoordinateMap> map_sptr;
  auto scannerBlocks_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_sptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_sptr->set_up();

  VectorWithOffset<int> num_axial_pos_per_segment(scannerBlocks_sptr->get_num_rings() * 2 - 1);
  VectorWithOffset<int> min_ring_diff_v(scannerBlocks_sptr->get_num_rings() * 2 - 1);
  VectorWithOffset<int> max_ring_diff_v(scannerBlocks_sptr->get_num_rings() * 2 - 1);
  auto proj_data_info_blocks_sptr = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info_blocks_sptr = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_sptr);
  Bin bin, bin1, bin2, binR1;
  CartesianCoordinate3D<float> b1, b2, rb1, rb2;
  DetectionPosition<> det_pos, det_pos_ord;
  DetectionPositionPair<> dp1, dp2, dpR1;
  CartesianCoordinate3D<float> coord_ord;
  map_sptr = scannerBlocks_sptr->get_detector_map_sptr();
  int rad_size = map_sptr->get_num_radial_coords();
  int ax_size = map_sptr->get_num_axial_coords();
  int tang_size = map_sptr->get_num_tangential_coords();

  DetectorCoordinateMap::det_pos_to_coord_type coord_map_reordered;

  //    reorder the tangential positions
  for (int rad = 0; rad < rad_size; rad++)
    for (int ax = 0; ax < ax_size; ax++)
      for (int tang = 0; tang < tang_size; tang++)
        {
          det_pos.radial_coord() = rad;
          det_pos.axial_coord() = ax;
          det_pos.tangential_coord() = tang;
          det_pos_ord.radial_coord() = rad;
          det_pos_ord.axial_coord() = ax;
          det_pos_ord.tangential_coord() = tang_size - 1 - tang;

          coord_ord = map_sptr->get_coordinate_for_det_pos(det_pos_ord);
          coord_map_reordered[det_pos] = coord_ord;
        }

  auto scannerBlocks_reord_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_reord_sptr->set_scanner_geometry("Generic");
  //    scannerBlocks_reord_sptr->set_num_transaxial_blocks_per_bucket(1);
  scannerBlocks_reord_sptr->set_detector_map(coord_map_reordered);
  scannerBlocks_reord_sptr->set_up();

  auto proj_data_info_blocks_reord_sptr = std::make_shared<ProjDataInfoGenericNoArcCorr>();
  proj_data_info_blocks_reord_sptr = set_blocks_projdata_info<ProjDataInfoGenericNoArcCorr>(scannerBlocks_reord_sptr);
  timer.reset();
  timer.start();

  //    now forward-project images

  auto PM = std::make_shared<ProjMatrixByBinUsingRayTracing>();
  PM->enable_cache(false);
  auto forw_projector1_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  info(boost::format("Test blocks on Cylindrical: Forward projector used: %1%") % forw_projector1_sptr->parameter_info());
  forw_projector1_sptr->set_up(proj_data_info_blocks_sptr, image1_sptr);

  auto forw_projector2_sptr = std::make_shared<ForwardProjectorByBinUsingProjMatrixByBin>(PM);
  forw_projector2_sptr->set_up(proj_data_info_blocks_reord_sptr, image1_sptr);

  auto projdata1 = std::make_shared<ProjDataInMemory>(exam_info_sptr,
                                                      proj_data_info_blocks_sptr); //,
  //                                             "sino1_map.hs",std::ios::out | std::ios::trunc | std::ios::in));

  auto projdata2 = std::make_shared<ProjDataInMemory>(exam_info_sptr,
                                                      proj_data_info_blocks_reord_sptr); //,
  //                                             "sino2_map.hs",std::ios::out | std::ios::trunc | std::ios::in));

  forw_projector1_sptr->forward_project(*projdata1, *image1_sptr);
  forw_projector2_sptr->forward_project(*projdata2, *image1_sptr);

  for (int view = 0; view <= proj_data_info_blocks_reord_sptr->get_max_view_num(); view++)
    {

      bin.segment_num() = 0;
      bin.axial_pos_num() = 0;
      bin.view_num() = view;
      bin.tangential_pos_num() = 0;

      proj_data_info_blocks_sptr->get_det_pos_pair_for_bin(dp1, bin);
      proj_data_info_blocks_reord_sptr->get_det_pos_pair_for_bin(dp2, bin);

      proj_data_info_blocks_sptr->get_bin_for_det_pos_pair(bin1, dp1);
      proj_data_info_blocks_reord_sptr->get_bin_for_det_pos_pair(bin2, dp2);

      //        //                check cartesian coordinates of detectors
      proj_data_info_blocks_sptr->find_cartesian_coordinates_of_detection(b1, b2, bin1);
      proj_data_info_blocks_reord_sptr->find_cartesian_coordinates_of_detection(rb1, rb2, bin1);

      //        now get det_pos from the reordered coord ir shouls be different from the one obtained for bin and bin1
      proj_data_info_blocks_sptr->find_bin_given_cartesian_coordinates_of_detection(binR1, rb1, rb2);
      proj_data_info_blocks_sptr->get_det_pos_pair_for_bin(dpR1, binR1);

      check_if_equal(projdata1->get_bin_value(bin1), projdata2->get_bin_value(bin2),
                     " checking cartesian coordinate y1 are the same on a flat bucket");
      check(b1 != rb1, " checking cartesian coordinate of detector 1 are different if  we use a reordered map");
      check(b2 != rb2, " checking cartesian coordinate of detector 2 are different if  we use a reordered map");
      check(dp1.pos1().tangential_coord() != dpR1.pos1().tangential_coord(),
            " checking det_pos.tangential is different if we use a reordered map");
    }
  timer.stop();
  std::cerr << "-- CPU Time " << timer.value() << '\n';
}

void
BlocksTests::run_intersection_with_cylinder_test()
{
  //    create projadata info
  auto scannerBlocks_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scannerBlocks_sptr->set_average_depth_of_interaction(5);
  scannerBlocks_sptr->set_num_axial_crystals_per_block(1);
  scannerBlocks_sptr->set_axial_block_spacing(scannerBlocks_sptr->get_axial_crystal_spacing()
                                              * scannerBlocks_sptr->get_num_axial_crystals_per_block());
  scannerBlocks_sptr->set_transaxial_block_spacing(scannerBlocks_sptr->get_transaxial_crystal_spacing()
                                                   * scannerBlocks_sptr->get_num_transaxial_crystals_per_block());
  scannerBlocks_sptr->set_num_axial_blocks_per_bucket(2);
  scannerBlocks_sptr->set_num_rings(2);
  scannerBlocks_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scannerBlocks_sptr->set_up();

  auto proj_data_info = std::make_shared<ProjDataInfoBlocksOnCylindricalNoArcCorr>();
  proj_data_info = set_blocks_projdata_info<ProjDataInfoBlocksOnCylindricalNoArcCorr>(scannerBlocks_sptr);

  // loop over all LORs in the projdata
  Bin bin;
  LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;
  LORAs2Points<float> lor_points, lor_points_with_larger_radius;
  const float radius = proj_data_info->get_scanner_sptr()->get_inner_ring_radius();

  auto are_parallel = [](CartesianCoordinate3D<float> &line1, CartesianCoordinate3D<float> &line2) -> bool {
    const auto dot_product = line1.x() * line2.x() + line1.y() * line2.y() + line1.z() * line2.z();
    const auto length1 = sqrt(line1.x() * line1.x() + line1.y() * line1.y() + line1.z() * line1.z());
    const auto length2 = sqrt(line2.x() * line2.x() + line2.y() * line2.y() + line2.z() * line2.z());
    return abs(dot_product) / length1 / length2 > 0.99;
  };

  auto point_is_on_cylinder = [](const CartesianCoordinate3D<float> &point, float radius) -> bool {
    return abs(sqrt(point.x() * point.x() + point.y() * point.y()) - radius) < 0.1;
  };

  const auto segment_sequence = ProjData::standard_segment_sequence(*proj_data_info);
  std::size_t index(0);
  for (int seg : segment_sequence)
  {
    bin.segment_num() = seg;
    for (bin.axial_pos_num() = proj_data_info->get_min_axial_pos_num(bin.segment_num());
          bin.axial_pos_num() <= proj_data_info->get_max_axial_pos_num(bin.segment_num()); ++bin.axial_pos_num())
    {
      for (bin.view_num() = proj_data_info->get_min_view_num(); bin.view_num() <= proj_data_info->get_max_view_num();
            ++bin.view_num())
      {
        for (bin.tangential_pos_num() = proj_data_info->get_min_tangential_pos_num();
              bin.tangential_pos_num() <= proj_data_info->get_max_tangential_pos_num(); ++bin.tangential_pos_num())
        {
          proj_data_info->get_LOR(lor, bin);
          if (lor.get_intersections_with_cylinder(lor_points, radius) == Succeeded::yes &&
              lor.get_intersections_with_cylinder(lor_points_with_larger_radius, radius + 20) == Succeeded::yes)
          {
            const CartesianCoordinate3D<float> p1 = lor_points.p1();
            const CartesianCoordinate3D<float> p2 = lor_points.p2();
            const CartesianCoordinate3D<float> p1_large_r = lor_points_with_larger_radius.p1();
            const CartesianCoordinate3D<float> p2_large_r = lor_points_with_larger_radius.p2();

            // check if on same line
            auto v1 = CartesianCoordinate3D<float>(p1 - p2);
            auto v2 = CartesianCoordinate3D<float>(p1 - p1_large_r);
            auto v3 = CartesianCoordinate3D<float>(p1 - p2_large_r);
            check(are_parallel(v1, v2), "checking for collinearity");
            check(are_parallel(v2, v3), "checking for collinearity");

            // check if on cylinder
            check(point_is_on_cylinder(p1, radius), "check if point is on cylinder");
            check(point_is_on_cylinder(p2, radius), "check if point is on cylinder");
            check(point_is_on_cylinder(p1_large_r, radius + 20), "check if point is on cylinder");
            check(point_is_on_cylinder(p2_large_r, radius + 20), "check if point is on cylinder");
          }
        }
      }
    }
  }
}

void
BlocksTests::run_tests()
{

  std::cerr << "-------- Testing Blocks Geometry --------\n";
  run_voxelOnCartesianGrid_with_negative_offset();
  run_axial_projection_test();
  run_map_orientation_test();
  run_symmetry_test();
  run_plane_symmetry_test();
  run_intersection_with_cylinder_test();
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  Verbosity::set(1);
  BlocksTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
