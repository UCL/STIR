/*
    Copyright (C) 2017, 2022, UCL
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
  \ingroup recon_test
  Implementation of stir::test_view_offset_root
*/
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/centre_of_gravity.h"
#include "stir/listmode/CListModeDataROOT.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/read_from_file.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/RunTests.h"
#include "boost/lexical_cast.hpp"

#include "stir/warning.h"

#include <stack>

using std::cerr;
using std::ifstream;

START_NAMESPACE_STIR

/*!
 *  \ingroup recon_test
 *  \brief Test class to check the consistency between ROOT listmode and STIR backprojection
 *  \author Elise Emond
 *  \author Robert Twyman
 *
 * This test currently uses Root listmodes of single point sources.
 * This test computes the distance between the original point source position and the closes voxel that the list mode
 * event passes through. Ideally each event would travel directly through the original point source position but
 * error may be present. Therefore we test that the majority of LORs travel close enough.
 */

class ROOTconsistency_Tests : public RunTests
{
public:
  ROOTconsistency_Tests(const std::string& in, const std::string& image)
    : root_header_filename(in), image_filename(image)
    {}
    void run_tests();

private:
    /*! Reads listmode event by event, computes the ProjMatrixElemsForOneBin (probabilities
     * along a bin LOR). Passes ProjMatrixElemsForOneBin (LOR) to test_LOR_closest_approach() and if fails,
     * add 1 to `failed_events` (LOR's closest voxel was not within tolerance).
     * Check if the number of `failed_events` is greater than half the number of tested events to pass the test.
     * @param test_discretised_density_sptr Density containing a point source.
     * @param original_coords Precomputed coordinates of the point source
     * @param grid_spacing Precomputed voxel sizes
     */
    void test_lm_data_closest_approaches(
            const shared_ptr <DiscretisedDensity<3, float>> &test_discretised_density_sptr,
            const CartesianCoordinate3D<float> original_coords, CartesianCoordinate3D<float> grid_spacing);

    /*! Given a ProjMatrixElemsForOneBin (probabilities), test if the closest voxel in the LOR to the original_coords
     * is within tolerance distance. If it is, pass with true, otherwise fales.
     * @param probabilities ProjMatrixElemsForOneBin object of a list mode event
     * @param test_discretised_density_sptr Density containing a point source.
     * @param original_coords Precomputed coordinates of the point source
     * @param grid_spacing Precomputed voxel sizes - used in setting tolerance
     * @return True if test passes, false if failed.
     */
    bool test_LOR_closest_approach(const ProjMatrixElemsForOneBin &probabilities,
                                   const shared_ptr <DiscretisedDensity<3, float>> &test_discretised_density_sptr,
                                   const CartesianCoordinate3D<float> original_coords,
                                   const float grid_spacing);

	std::string root_header_filename;
	std::string image_filename;
};

void
ROOTconsistency_Tests::run_tests()
{
  // DiscretisedDensity for original image
  shared_ptr<DiscretisedDensity<3, float> > discretised_density_sptr(DiscretisedDensity<3,float>::read_from_file(image_filename));

  // needs to be cast to VoxelsOnCartesianGrid to be able to calculate the centre of gravity,
  // hence the location of the original source, stored in test_original_coords.
  const VoxelsOnCartesianGrid<float>& discretised_cartesian_grid =
      dynamic_cast<VoxelsOnCartesianGrid<float> &>(*discretised_density_sptr);

  // Find the center of mass of the original data
  CartesianCoordinate3D<float> original_coords = find_centre_of_gravity_in_mm(discretised_cartesian_grid);
  CartesianCoordinate3D<float> grid_spacing = discretised_cartesian_grid.get_grid_spacing();

  test_lm_data_closest_approaches(discretised_density_sptr, original_coords, grid_spacing);
}

void
ROOTconsistency_Tests::
test_lm_data_closest_approaches(
        const shared_ptr <DiscretisedDensity<3, float>> &test_discretised_density_sptr,
        const CartesianCoordinate3D<float> original_coords, CartesianCoordinate3D<float> grid_spacing)
{
  shared_ptr<CListModeData> lm_data_sptr(read_from_file<CListModeData>(root_header_filename));

  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());

  proj_matrix_sptr.get()->set_up(lm_data_sptr->get_proj_data_info_sptr(),
                                 test_discretised_density_sptr);

  ProjMatrixElemsForOneBin proj_matrix_row;

  // The number of LORs with closes approach greater than the threshold.
  int failed_events = 0;
  int tested_events = 0;
  const auto tolerance = 1.5 * static_cast<float>(norm(grid_spacing)); // Using norm(grid_spacing) as a tolerance

  {
    // loop over all events in the listmode file
    shared_ptr <CListRecord> record_sptr = lm_data_sptr->get_empty_record_sptr();
    CListRecord& record = *record_sptr;
    while (lm_data_sptr->get_next_record(record) == Succeeded::yes)
    {
      // only stores prompts
        if (record.is_event() && record.event().is_prompt())
        {
          Bin bin;
          bin.set_bin_value(1.f);
          // gets the bin corresponding to the event
          record.event().get_bin(bin, *lm_data_sptr->get_proj_data_info_sptr());
          if ( bin.get_bin_value()>0 )
          {
            // computes the TOF probabilities along the bin LOR
            proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, bin);
            // adds coordinates and weights of the elements with highest probability along LOR
            if (!test_LOR_closest_approach(proj_matrix_row, test_discretised_density_sptr,
                                           original_coords, tolerance))
            {
              failed_events += 1;
            }
            tested_events += 1;
          }
        }
      }
    }

  cerr << "\nNUMBER OF FAILED EVENTS = " << failed_events << "\t NUMBER OF TESTED EVENTS = " << tested_events << std::endl;
  check_if_less(failed_events, 0.5 * tested_events,
                "the number of failed events is more than half the number of tested events.");

}

bool
ROOTconsistency_Tests::
test_LOR_closest_approach(const ProjMatrixElemsForOneBin &probabilities,
                          const shared_ptr <DiscretisedDensity<3, float>> &test_discretised_density_sptr,
                          const CartesianCoordinate3D<float> original_coords,
                          const float tolerance)
{
  // Loop variables
  CartesianCoordinate3D<float> closest_LOR_voxel_to_origin;
  float min_distance;
  bool first_entry = true; // Use this to populate with the initial value

  ProjMatrixElemsForOneBin::const_iterator element_ptr = probabilities.begin();
  // iterate over all to element_ptr to find the minimal distance between LOR and original_coords
  while (element_ptr != probabilities.end())
  {
    CartesianCoordinate3D<float> voxel_coords =
            test_discretised_density_sptr->get_physical_coordinates_for_indices(element_ptr->get_coords());

    float dist_to_original = norm(voxel_coords - original_coords);
    if (dist_to_original < min_distance || first_entry)
    {
      closest_LOR_voxel_to_origin = voxel_coords;
      min_distance = dist_to_original;
      first_entry = false;
    }
    ++element_ptr;
  }
//  if (!check_if_less(min_distance, 3*norm(grid_spacing), "ERR msgs"))
  if (min_distance > tolerance)
  {
//    cerr << "min_distance = " << min_distance << "\t tolerance = " << tolerance << std::endl;
    return false; // Test failed - LOR closest voxel beyond tolerance
  }
  return true; // Test passed - LOR closest voxel within tolerance
}

END_NAMESPACE_STIR

int main(int argc, char **argv)
{
  USING_NAMESPACE_STIR

  if (argc != 3)
  {
    cerr << "Usage : " << argv[1] << " filename "
         << argv[2] << "original image \n"
         << "See source file for the format of this file.\n\n";
    return EXIT_FAILURE;
  }


  ifstream in(argv[1]);
  if (!in)
  {
    cerr << argv[0]
         << ": Error opening root file " << argv[1] << "\nExiting.\n";

    return EXIT_FAILURE;
  }
  ifstream in2(argv[2]);
  if (!in2)
  {
    cerr << argv[0]
         << ": Error opening original image " << argv[2] << "\nExiting.\n";

    return EXIT_FAILURE;
  }
  std::string filename(argv[1]);
  std::string image(argv[2]);
    ROOTconsistency_Tests tests(filename,image);
    tests.run_tests();
    return tests.main_return_value();
}