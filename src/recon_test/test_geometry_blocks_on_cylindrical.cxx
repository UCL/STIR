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
#include "stir/format.h"
#include <cmath>

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
  /*! \brief Tests multiple axial blocks/bucket configurations to ensure the detector map's axial indices and coordinates
   * are monotonic
   */
  void run_monotonic_axial_coordinates_in_detector_map_test();
  //! Tests the axial indices and coordinates are monotonic in the detector map
  static Succeeded monotonic_axial_coordinates_in_detector_map_test(const shared_ptr<Scanner>& scanner_sptr);
};

void
GeometryBlocksOnCylindricalTests::run_monotonic_axial_coordinates_in_detector_map_test()
{
  auto scanner_sptr = std::make_shared<Scanner>(Scanner::SAFIRDualRingPrototype);
  scanner_sptr->set_scanner_geometry("BlocksOnCylindrical");
  scanner_sptr->set_transaxial_block_spacing(scanner_sptr->get_transaxial_crystal_spacing()
                                             * scanner_sptr->get_num_transaxial_crystals_per_block());
  int num_axial_buckets = 1; // TODO add for loop when support is added

  for (int num_axial_crystals_per_blocks = 1; num_axial_crystals_per_blocks < 3; ++num_axial_crystals_per_blocks)
    for (int num_axial_blocks_per_bucket = 1; num_axial_blocks_per_bucket < 3; ++num_axial_blocks_per_bucket)
      {
        scanner_sptr->set_num_axial_crystals_per_block(num_axial_crystals_per_blocks);
        scanner_sptr->set_num_axial_blocks_per_bucket(num_axial_blocks_per_bucket);
        scanner_sptr->set_num_rings(scanner_sptr->get_num_axial_crystals_per_bucket() * num_axial_buckets);
        scanner_sptr->set_axial_block_spacing(scanner_sptr->get_axial_crystal_spacing()
                                              * (scanner_sptr->get_num_axial_crystals_per_block() + 0.5));

        if (monotonic_axial_coordinates_in_detector_map_test(scanner_sptr) == Succeeded::no)
          {
            warning(format("Monothonic axial coordinates test failed for:\n"
                           "\taxial_crystal_per_block =\t{}\n"
                           "\taxial_blocks_per_bucket =\t{}\n"
                           "\tnum_axial_buckets =\t\t\t{}",
                           num_axial_crystals_per_blocks,
                           num_axial_blocks_per_bucket,
                           num_axial_buckets));
            everything_ok = false;
            return;
          }
      }
}

Succeeded
GeometryBlocksOnCylindricalTests::monotonic_axial_coordinates_in_detector_map_test(const shared_ptr<Scanner>& scanner_sptr)
{
  if (scanner_sptr->get_scanner_geometry() != "BlocksOnCylindrical")
    {
      warning("monotonic_axial_coordinates_in_detector_map_test is only for the BlocksOnCylindrical geometry");
      return Succeeded::no;
    }

  shared_ptr<DetectorCoordinateMap> detector_map_sptr;
  try
    {
      detector_map_sptr.reset(new GeometryBlocksOnCylindrical(*scanner_sptr));
    }
  catch (const std::runtime_error& e)
    {
      warning(format("Caught runtime_error while creating GeometryBlocksOnCylindrical: {}\n"
                     "Failing the test.",
                     e.what()));
      return Succeeded::no;
    }

  unsigned min_axial_pos = 0;
  float prev_min_axial_coord = -std::numeric_limits<float>::max();

  for (unsigned axial_idx = 0; axial_idx < detector_map_sptr->get_num_axial_coords(); ++axial_idx)
    for (unsigned tangential_idx = 0; tangential_idx < detector_map_sptr->get_num_tangential_coords(); ++tangential_idx)
      for (unsigned radial_idx = 0; radial_idx < detector_map_sptr->get_num_radial_coords(); ++radial_idx)
        {
          const DetectionPosition<> det_pos = DetectionPosition<>(tangential_idx, axial_idx, radial_idx);
          CartesianCoordinate3D<float> coord = detector_map_sptr->get_coordinate_for_det_pos(det_pos);
          if (coord.z() > prev_min_axial_coord)
            {
              min_axial_pos = axial_idx;
              prev_min_axial_coord = coord.z();
            }
          else if (coord.z() < prev_min_axial_coord)
            {
              float delta = coord.z() - prev_min_axial_coord;
              warning(format("Axial Coordinates are not monotonic.\n"
                             "Next axial index =\t\t{}, Next axial coord (mm) =\t\t{}  ({})\n"
                             "Previous axial index =\t{}, Previous axial coord (mm) =\t{}",
                             axial_idx,
                             coord.z(),
                             delta,
                             min_axial_pos,
                             prev_min_axial_coord));
              return Succeeded::no;
            }
        }

  return Succeeded::yes;
}

void
GeometryBlocksOnCylindricalTests::run_tests()
{
  HighResWallClockTimer timer;
  timer.start();
  run_monotonic_axial_coordinates_in_detector_map_test();
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