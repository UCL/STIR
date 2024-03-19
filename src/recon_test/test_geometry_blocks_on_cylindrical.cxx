/*!

\file
\ingroup recontest

\brief Test program to ensure the axial coordinates of blocks on cylindrical are monotonic with axial indices

\author Robert Twyman

    Copyright (C) 2024, Prescient Imaging
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/ExamInfo.h"
#include "stir/Verbosity.h"
#include "stir/LORCoordinates.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "stir/Scanner.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/GeometryBlocksOnCylindrical.h"
#include "stir/IO/write_to_file.h"
#include <cmath>
// #include <filesystem>

START_NAMESPACE_STIR

/*!
\ingroup test
\brief Test class for BlocksOnCylindrical geometry
*/
class GeometryBlocksOnCylindricalTests : public RunTests
{
public:
  void run_tests() override;

private:
  //! Loop through all transaxial and axial indices to ensure the coordinates are monotonic, indicating a spiralling
  // generation of the coordinates of the coordinates around the scanner geometry
  void run_monotonic_coordinates_generation_test();
  /*! \brief Tests multiple axial blocks/bucket configurations to ensure the detector map's axial indices and coordinates
   * are monotonic */
  void run_monotonic_axial_coordinates_in_detector_map_test();
  //! Tests the axial indices and coordinates are monotonic in the detector map
  static Succeeded monotonic_axial_coordinates_in_detector_map_test(const shared_ptr<Scanner>& scanner_sptr);

  //! Sets up the scanner for the test
  static Succeeded setup_scanner_for_test(shared_ptr<stir::Scanner> scanner_sptr);

  void run_assert_scanner_centred_on_origin_test();

  /*! This is a test of the start_z position. The code was refactored and this test ensures that the new calculation is
   * equivalent to the old calculation */
  void validate_start_z_with_old_calculation();

  void validate_first_bucket_is_centred_on_x_axis();
  ;
};

void
GeometryBlocksOnCylindricalTests::run_monotonic_axial_coordinates_in_detector_map_test()
{
  auto scanner_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");

  //  for (int num_axial_crystals_per_blocks = 1; num_axial_crystals_per_blocks < 3; ++num_axial_crystals_per_blocks)
  //    for (int num_axial_blocks_per_bucket = 1; num_axial_blocks_per_bucket < 3; ++num_axial_blocks_per_bucket)
  //      for (int num_axial_buckets = 1; num_axial_buckets < 3; ++num_axial_buckets)

  // TESTING CONFIG:
  int num_axial_buckets = 2;
  int num_axial_blocks_per_bucket = 2;
  int num_axial_crystals_per_blocks = 2;
  {
    int num_rings = num_axial_crystals_per_blocks * num_axial_blocks_per_bucket * num_axial_buckets;
    scanner_sptr->set_num_axial_crystals_per_block(num_axial_crystals_per_blocks);
    scanner_sptr->set_num_axial_blocks_per_bucket(num_axial_blocks_per_bucket);
    scanner_sptr->set_num_rings(num_rings);

    scanner_sptr->set_axial_block_spacing(scanner_sptr->get_axial_crystal_spacing()
                                          * (scanner_sptr->get_num_axial_crystals_per_block() + 0.5));

    //    if (num_axial_buckets > 1)
    //      scanner_sptr->set_axial_bucket_spacing(scanner_sptr->get_axial_block_spacing() * 1.5);
    //    else
    //      scanner_sptr->set_axial_bucket_spacing(-1);

    if (monotonic_axial_coordinates_in_detector_map_test(scanner_sptr) == Succeeded::no)
      {
        warning(boost::format("Monothonic axial coordinates test failed for:\n"
                              "\taxial_crystal_per_block =\t%1%\n"
                              "\taxial_blocks_per_bucket =\t%2%\n"
                              "\tnum_axial_buckets =\t\t\t%3%")
                % num_axial_crystals_per_blocks % num_axial_blocks_per_bucket % scanner_sptr->get_num_axial_buckets());
        everything_ok = false;
        return;
      }
  }
}

Succeeded
GeometryBlocksOnCylindricalTests::setup_scanner_for_test(shared_ptr<stir::Scanner> scanner_sptr)
{
  if (scanner_sptr->get_scanner_geometry() != "BlocksOnCylindrical")
    {
      warning("monotonic_axial_coordinates_in_detector_map_test is only for the BlocksOnCylindrical geometry");
      return Succeeded::no;
    }

  try
    {
      scanner_sptr->set_up();
    }
  catch (const std::runtime_error& e)
    {
      warning(boost::format("Caught runtime_error while creating GeometryBlocksOnCylindrical: %1%\n"
                            "Failing the test.")
              % e.what());
      return Succeeded::no;
    }
  return Succeeded::yes;
}

Succeeded
GeometryBlocksOnCylindricalTests::monotonic_axial_coordinates_in_detector_map_test(const shared_ptr<Scanner>& scanner_sptr)
{
  if (setup_scanner_for_test(scanner_sptr) == Succeeded::no)
    return Succeeded::no;

  unsigned min_axial_pos = 0;
  float prev_min_axial_coord = -std::numeric_limits<float>::max();
  shared_ptr<const DetectorCoordinateMap> detector_map_sptr = scanner_sptr->get_detector_map_sptr();

  for (unsigned axial_idx = 0; axial_idx < detector_map_sptr->get_num_axial_coords(); ++axial_idx)
    {
      const DetectionPosition<> det_pos = DetectionPosition<>(0, axial_idx, 0);
      CartesianCoordinate3D<float> coord = detector_map_sptr->get_coordinate_for_det_pos(det_pos);
      //          std::cerr << "coord.z() = " << coord.z() << "\tprev_min_axial_coord = " << prev_min_axial_coord
      //                    << "\tdelta = " << coord.z() - prev_min_axial_coord << std::endl;
      if (coord.z() > prev_min_axial_coord)
        {
          min_axial_pos = axial_idx;
          prev_min_axial_coord = coord.z();
        }
      else if (coord.z() < prev_min_axial_coord)
        {
          float delta = coord.z() - prev_min_axial_coord;
          warning(boost::format("Axial Coordinates are not monotonic.\n"
                                "Next axial index =\t\t%1%, Next axial coord (mm) =\t\t%2%  (%3%)\n"
                                "Previous axial index =\t%4%, Previous axial coord (mm) =\t%5%")
                  % axial_idx % coord.z() % delta % min_axial_pos % prev_min_axial_coord);
          return Succeeded::no;
        }
    }

  return Succeeded::yes;
}

void
GeometryBlocksOnCylindricalTests::run_assert_scanner_centred_on_origin_test()
{
  //  auto scanner_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  //  scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");
  //  if (setup_scanner_for_test(scanner_sptr) == Succeeded::no)
  //    {
  //      warning("GeometryBlocksOnCylindricalTests::run_assert_scanner_centred_on_origin_test: "
  //              "Scanner not set up correctly for test");
  //      everything_ok = false;
  //    }
  //
  //  auto detector_map_sptr = scanner_sptr->get_detector_map_sptr();
  //  for (int transaxial_idx = 0; transaxial_idx < scanner_sptr->get_num_detectors_per_ring(); ++transaxial_idx)
  //    {
  //      const DetectionPosition<> det_pos = DetectionPosition<>(transaxial_idx, 0, 0);
  //      auto cart_coord = detector_map_sptr->get_coordinate_for_det_pos(det_pos);
  //      if (cart_coord.z() != 0.0)
  //        {
  //          warning(boost::format("Scanner Z component of the first index is > 0 \n"
  //                                "Transaxial index =\t%1%\n"
  //                                "Axial index =\t%2%\n"
  //                                "Cartesian Coordinate =\t%2%")
  //                  % transaxial_idx % cart_coord);
  //          everything_ok = false;
  //        }
  //    }
}

void
GeometryBlocksOnCylindricalTests::run_monotonic_coordinates_generation_test()
{
  auto scanner_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");

  int prev_max_axial_coord = -1;
  int prev_max_transaxial_coord = -1;

  for (int ax_bucket_num = 0; ax_bucket_num < scanner_sptr->get_num_axial_buckets(); ++ax_bucket_num)
    for (int ax_block_num = 0; ax_block_num < scanner_sptr->get_num_axial_blocks_per_bucket(); ++ax_block_num)
      for (int ax_crystal_num = 0; ax_crystal_num < scanner_sptr->get_num_axial_crystals_per_block(); ++ax_crystal_num)
        for (int trans_bucket_num = 0; trans_bucket_num < scanner_sptr->get_num_transaxial_buckets(); ++trans_bucket_num)
          for (int trans_block_num = 0; trans_block_num < scanner_sptr->get_num_transaxial_blocks_per_bucket(); ++trans_block_num)
            for (int trans_crys_num = 0; trans_crys_num < scanner_sptr->get_num_transaxial_crystals_per_block(); ++trans_crys_num)
              {
                int axial_coord
                    = GeometryBlocksOnCylindrical::get_axial_coord(*scanner_sptr, ax_bucket_num, ax_block_num, ax_crystal_num);
                int transaxial_coord;
                transaxial_coord = GeometryBlocksOnCylindrical::get_transaxial_coord(
                    *scanner_sptr, trans_bucket_num, trans_block_num, trans_crys_num);

                // Test that the axial coordinates are monotonic
                if (prev_max_axial_coord > axial_coord)
                  {
                    warning(boost::format("Axial Coordinates are not monotonic.\n"
                                          "Next axial index =\t\t%1%\n"
                                          "Previous axial index =\t%2%\n"
                                          "Next Transaxial index =\t%3%\n"
                                          "Previous Transaxial index =\t%4%")
                            % axial_coord % prev_max_axial_coord % transaxial_coord % prev_max_transaxial_coord);
                    everything_ok = false;
                    return;
                  }

                // Test that the transaxial coordinates are monotonic, it will reset to 0 when the axial index increases
                if (prev_max_transaxial_coord > transaxial_coord && axial_coord <= prev_max_axial_coord)
                  {
                    warning(boost::format("Transaxial Coordinates are not monotonic.\n"
                                          "Next axial index =\t\t%1%\n"
                                          "Previous axial index =\t%2%\n"
                                          "Next Transaxial index =\t%3%\n"
                                          "Previous Transaxial index =\t%4%")
                            % axial_coord % prev_max_axial_coord % transaxial_coord % prev_max_transaxial_coord);
                    everything_ok = false;
                    return;
                  }
              }
}

void
GeometryBlocksOnCylindricalTests::validate_start_z_with_old_calculation()
{
  auto scanner_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");

  // calculate the start_z with the old equation.
  // The code was moved here and slightly refactored to get values from the scanner object
  float axial_blocks_gap = scanner_sptr->get_axial_block_spacing()
                           - scanner_sptr->get_num_axial_crystals_per_block() * scanner_sptr->get_axial_crystal_spacing();
  float old_code_calculation
      = -(scanner_sptr->get_axial_block_spacing() * (scanner_sptr->get_num_axial_blocks_per_bucket())
              * scanner_sptr->get_num_axial_buckets()
          - scanner_sptr->get_axial_crystal_spacing()
          - axial_blocks_gap * (scanner_sptr->get_num_axial_blocks_per_bucket() * scanner_sptr->get_num_axial_buckets() - 1))
        / 2;

  // new method to calculate start_z
  float start_z_method = GeometryBlocksOnCylindrical::get_initial_axial_z_offset(*scanner_sptr);

  if (std::abs(old_code_calculation - start_z_method) > 1e-6)
    {
      warning(boost::format("Old code calculation =\t%1%\n"
                            "New code calculation =\t%2%\n"
                            "Difference =\t%3%")
              % old_code_calculation % start_z_method % (old_code_calculation - start_z_method));
      everything_ok = false;
    }
}

void
GeometryBlocksOnCylindricalTests::validate_first_bucket_is_centred_on_x_axis()
{
  auto scanner_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scanner_sptr->set_num_transaxial_blocks_per_bucket(1);
  scanner_sptr->set_intrinsic_azimuthal_tilt(-0.235307813); // Required by test calculations

  GeometryBlocksOnCylindrical detector_map_builder(*scanner_sptr);
  scanner_sptr->get_num_transaxial_blocks_per_bucket();
  int first_transaxial_index_in_first_bucket = detector_map_builder.get_transaxial_coord(*scanner_sptr, 0, 0, 0);
  int last_transaxial_index_in_first_bucket
      = detector_map_builder.get_transaxial_coord(*scanner_sptr,
                                                  0,
                                                  scanner_sptr->get_num_transaxial_blocks_per_bucket() - 1,
                                                  scanner_sptr->get_num_transaxial_crystals_per_block());
  auto first_transaxial_coord_in_first_bucket
      = detector_map_builder.get_coordinate_for_det_pos(DetectionPosition<>(first_transaxial_index_in_first_bucket, 0, 0));
  auto last_transaxial_coord_in_first_bucket
      = detector_map_builder.get_coordinate_for_det_pos(DetectionPosition<>(last_transaxial_index_in_first_bucket, 0, 0));

  { // Testing the coordinates of the first axial row of the first bucket.
    // For this scanner, with the set intrinsic_azimuthal_tilt, the first bucket's tangential coordinates should
    // be centered at x=0 with a constant z,y position

    // Get the coordinates of the first block/bucket
    std::vector<CartesianCoordinate3D<float>> crystal_coords(scanner_sptr->get_num_transaxial_crystals_per_bucket());
    for (int i = 0; i < crystal_coords.size(); i++)
      {
        crystal_coords[i] = detector_map_builder.get_coordinate_for_det_pos(DetectionPosition<>(i, 0, 0));
      }

    {
      float previous_x = -std::numeric_limits<float>::max();
      for (int i = 1; i < crystal_coords.size(); i++)
        {
          // Check if the crystal coords are all at the same z,y position
          check_if_equal(crystal_coords[0].z(), crystal_coords[i].z());
          check_if_equal(crystal_coords[0].y(), crystal_coords[i].y());
          // Check the x position is monotonic
          check_if_less(previous_x,
                        crystal_coords[i].x(),
                        boost::str(boost::format("Crystal %1% x position is more than "
                                                 "the previous crystal x position")
                                   % i));
        }
    }

    // Check if x position is centered on 0.0
    for (int i = 0; i < int(crystal_coords.size() / 2); i++)
      {
        float left_crystal_x = crystal_coords[i].x();
        float right_crystal_x = crystal_coords[crystal_coords.size() - 1 - i].x();
        float offset_x = abs(left_crystal_x + right_crystal_x) / 2;
        float acceptable_error = (abs(left_crystal_x) + abs(right_crystal_x)) * 1e-6;
        check_if_less(offset_x,
                      acceptable_error,
                      boost::str(boost::format("Left and right crystal x positions are not evenly "
                                               "distributed over x axis.\tLeft crystal x = %1%\tRight crystal x = "
                                               "%2%\tOffset = %3%")
                                 % left_crystal_x % right_crystal_x % offset_x));
      }

    // Check the central crystal x position is centered on the x axis if there is an odd number of crystals
    if (crystal_coords.size() % 2 == 1)
      {
        float central_x = crystal_coords[crystal_coords.size() / 2].x();
        float acceptable_error = abs(crystal_coords[0].x()) * 1e-6;
        check_if_less(abs(central_x),
                      acceptable_error,
                      boost::str(boost::format("With an odd number of crystals, the central crystal x position is not "
                                               "centered on the x axis.\tCentral crystal x = %1%\t Acceptable error = %2%")
                                 % central_x % acceptable_error));
      }
  }

  //  { // Loop over all transaxial buckets, blocks and crystals and save the coordinates to a csv file with the format
  //    // tang,axial,radial,z,y,x
  //    std::ofstream file;
  //
  //    const char* filename = "crystal_coords.csv";
  //    file.open(filename);
  //    for (int i = 0; i < scanner_sptr->get_num_detectors_per_ring(); i++)
  //      {
  //        auto coord = detector_map_builder.get_coordinate_for_det_pos(DetectionPosition<>(i, 0, 0));
  //        file << i << ",0,0," << coord.z() << "," << coord.y() << "," << coord.x() << std::endl;
  //      }
  //    file.close();
  //    std::cerr << "Crystal coordinates written to " << std::filesystem::current_path() / filename << std::endl;
  //  }
}

void
GeometryBlocksOnCylindricalTests::run_tests()
{
  HighResWallClockTimer timer;
  timer.start();
  run_monotonic_coordinates_generation_test();
  run_monotonic_axial_coordinates_in_detector_map_test();
  run_assert_scanner_centred_on_origin_test();
  validate_start_z_with_old_calculation();
  validate_first_bucket_is_centred_on_x_axis();
  timer.stop();
}
END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main()
{
  Verbosity::set(1);
  GeometryBlocksOnCylindricalTests tests;
  tests.run_tests();
  return tests.main_return_value();
}