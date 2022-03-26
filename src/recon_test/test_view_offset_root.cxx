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
#include "stir/Shape/GenerateImage.h"
#include "boost/lexical_cast.hpp"

#include "stir/warning.h"

#include <stack>

using std::cerr;
using std::ifstream;

START_NAMESPACE_STIR


class ROOTconsistency_Tests : public RunTests
{
public:
  ROOTconsistency_Tests(const char * const in, const char * const generate_image_parameter_filename)
    : root_header_filename(in), generate_image_parameter_filename(generate_image_parameter_filename)
    {}
    void run_tests();

private:
    /*! Reads listmode event by event, computes the ProjMatrixElemsForOneBin (probabilities
     * along a bin LOR). This method should include all tests/ setup for additional test such that the list data is
     * read only once.
     *
     * Passes ProjMatrixElemsForOneBin (LOR) to test_LOR_closest_approach() and if fails,
     * add 1 to `failed_events` (LOR's closest voxel was not within tolerance).
     * Check if the number of `failed_events` is greater than half the number of tested events to pass the test.
     * @param test_discretised_density_sptr Density containing a point source.
     * @param original_coords Precomputed coordinates of the point source
     * @param grid_spacing Precomputed voxel sizes
     */
    void process_list_data(
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
    const char * const generate_image_parameter_filename;

    //! A vector to store the coordinates of the closest approach voxels.
    std::vector<CartesianCoordinate3D<float>> min_distances_coords_list;
};

void
ROOTconsistency_Tests::run_tests()
{
  // Create the point source (discretised_density_sptr) with GenerateImage.
  GenerateImage image_gen_application(this->generate_image_parameter_filename);
  image_gen_application.compute();
  shared_ptr<DiscretisedDensity<3, float> > discretised_density_sptr = image_gen_application.get_output_sptr();

  // needs to be cast to VoxelsOnCartesianGrid to be able to calculate the centre of gravity,
  // hence the location of the original source, stored in test_original_coords.
  const VoxelsOnCartesianGrid<float>& discretised_cartesian_grid =
      dynamic_cast<VoxelsOnCartesianGrid<float> &>(*discretised_density_sptr);

  // Find the center of mass of the original data
  CartesianCoordinate3D<float> original_coords = find_centre_of_gravity_in_mm(discretised_cartesian_grid);
  CartesianCoordinate3D<float> grid_spacing = discretised_cartesian_grid.get_grid_spacing();

  // Iterate through the list mode data and perform all needed operations.
  process_list_data(discretised_density_sptr, original_coords, grid_spacing);
}

void
ROOTconsistency_Tests::
process_list_data(
        const shared_ptr <DiscretisedDensity<3, float>> &test_discretised_density_sptr,
        const CartesianCoordinate3D<float> original_coords, CartesianCoordinate3D<float> grid_spacing)
{
  shared_ptr<CListModeData> lm_data_sptr(read_from_file<CListModeData>(root_header_filename));

  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());

  proj_matrix_sptr.get()->set_up(lm_data_sptr->get_proj_data_info_sptr(),
                                 test_discretised_density_sptr);

  ProjMatrixElemsForOneBin proj_matrix_row;

  // For debugging, create a list of the closest LOR coords. Also first entry is the original.
  min_distances_coords_list.empty();
  min_distances_coords_list.push_back(original_coords);

  // The number of LORs with closes approach greater than the threshold.
  int failed_events = 0;
  int tested_events = 0;
  const auto tolerance = 1.5 * static_cast<float>(norm(grid_spacing)); // Using norm(grid_spacing) as a tolerance
  cerr << "Tolerance is set to " << tolerance << std::endl;

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

  { // Save the closest coordinate for each LOR to file.
    std::string lor_pos_filename = root_header_filename.substr(0, root_header_filename.size() - 6) + "_lor_pos.txt";
    cerr << "\nSaving debug information as: " << lor_pos_filename <<
            "\n The first entry is the original coordinate position." << std::endl;
    std::ofstream myfile;
    myfile.open(lor_pos_filename.c_str());
    for (std::vector<CartesianCoordinate3D<float>>::iterator coord_entry = min_distances_coords_list.begin();
         coord_entry != min_distances_coords_list.end(); ++coord_entry)
        myfile << coord_entry->x() << " " << coord_entry->y() << " " << coord_entry->z() << std::endl;
    myfile.close();
  }
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
  // Log position in vector
  min_distances_coords_list.push_back(closest_LOR_voxel_to_origin);

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
    cerr << "Usage : hroot_filename generate_image_par_filename\n\n";
    return EXIT_FAILURE;
  }

  { // Check the program arguments exist
    ifstream in(argv[1]);
    if (!in)
    {
      cerr << argv[0]
           << ": Error opening root header file: " << argv[1] << "\nExiting.\n";

      return EXIT_FAILURE;
    }
    ifstream in2(argv[2]);
    if (!in2)
    {
      cerr << argv[0]
           << ": Error opening generate image parameter filename: " << argv[2] << "\nExiting.\n";

      return EXIT_FAILURE;
    }
  }

  // ROOT consistency tests
  ROOTconsistency_Tests tests(argv[1], argv[2]);
  tests.run_tests();
  return tests.main_return_value();
}
