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

using std::cerr;
using std::ifstream;

START_NAMESPACE_STIR


class WeightedCoordinate
// This class is used to store the coordinates and weight
{
public:
  WeightedCoordinate()
      : coord(CartesianCoordinate3D<float>(0.f,0.f,0.f)), value(0.f)
  {}
  WeightedCoordinate(const CartesianCoordinate3D<float>& voxel_centre_v, const float value_v)
      : coord(voxel_centre_v), value(value_v)
  {}

  //! Coordinate position
  CartesianCoordinate3D<float> coord;
  //! Weight value
  float value;
};


class ROOTConsistencyTests : public RunTests
{
public:
  ROOTConsistencyTests()= default;

  /*!
   * \brief Run the tests with ROOT data

   * \returns 0 if all tests passed, 1 otherwise
   */
  void run_tests() override;

private:

  void setup();
  /*! Reads listmode event by event, computes the ProjMatrixElemsForOneBin (probabilities
   * along a bin LOR). This method should include all tests/ setup for additional test such that the list data is
   * read only once.
   *
   * Passes ProjMatrixElemsForOneBin (LOR) to test_nonTOF_LOR_closest_approach() and if fails,
   * add 1 to `failed_events` (LOR's closest voxel was not within nonTOF_distance_threshold).
   * Check if the number of `failed_events` is greater than half the number of tested events to pass the test.
   */
  void process_list_data();

  void loop_through_list_events(shared_ptr<CListModeData> lm_data_sptr, const shared_ptr<ProjMatrixByBin>& proj_matrix_sptr,
                                const CartesianCoordinate3D<float> original_coords);
  /*! Given a ProjMatrixElemsForOneBin (probabilities), test if the closest voxel in the LOR to the original_coords
   * is within nonTOF_distance_threshold distance. If it is, pass with true, otherwise fales.
   * @param probabilities ProjMatrixElemsForOneBin object of a list mode event
   * @return True if test passes, false if failed.
   */
  bool test_nonTOF_LOR_closest_approach(const ProjMatrixElemsForOneBin& probabilities);
  void post_processing_nonTOF();


  //! Selects and stores the highest probability elements of ProjMatrixElemsForOneBin.
  bool test_TOF_max_lor_voxel(const ProjMatrixElemsForOneBin& probabilities);

  //! Checks if original and calculated coordinates are close enough and saves data to file.
  void post_processing_TOF();

  // Auxilary methods
  std::string get_root_header_filename() { return "pretest_output/root_header_test" + std::to_string(test_index) + ".hroot"; }
  std::string get_generate_image_par_filename() { return "SourceFiles/generate_image" + std::to_string(test_index) + ".par"; }


  ///// Class VARIABLES /////

  // Data set variables
  int test_index;
  int num_tests;

  // Test results storage. Records if each test failed or not. E.g.`test_index` result is stored in `test_results_nonTOF[test_index -1]`
  std::vector<bool> test_results_nonTOF;
  std::vector<bool> test_results_TOF;

  shared_ptr<DiscretisedDensity<3, float> > discretised_density_sptr;
  CartesianCoordinate3D<float> original_coords; // Stored in class because of dynamic_cast
  CartesianCoordinate3D<float> grid_spacing;    // Stored in class because of dynamic_cast

  int num_events_tested = 0;

  /// NON TOF VARIABLES
  //! A vector to store the coordinates of the closest approach voxels.
  const float failure_tolerance_nonTOF = 0.05;
  std::vector<CartesianCoordinate3D<float>> nonTOF_closest_voxels_list;
  int num_failed_nonTOF_lor_events = 0;
  double nonTOF_distance_threshold;

  /// TOF VARIABLES
  //! A copy of the scanner timing resolution in mm
  const float failure_tolerance_TOF = 0.05;
  std::vector<WeightedCoordinate> TOF_LOR_peak_value_coords;
  double TOF_distance_threshold;
  int num_failed_TOF_lor_events = 0;
};

void
ROOTConsistencyTests::run_tests()
{
  // Loop over each of the ROOT files in the test_data directory
  num_tests = 8;
  test_results_nonTOF = std::vector<bool> (num_tests, true);
  test_results_TOF = std::vector<bool> (num_tests, true);

  cerr << "Testing the view offset consistency between GATE/ROOT and STIR. \n";
  for (int i = 1; i <= num_tests; ++i)
  {
    test_index = i;
    cerr << "\nTesting dataset " << std::to_string(test_index) << "...\n";

    setup();
    process_list_data();

    cerr << "\nResults for dataset: " << std::to_string(test_index) << std::endl;
    post_processing_nonTOF();
    post_processing_TOF();
  }

  // Print results for easy readability
  cerr << "\nTest Results\n"
          "------------------------------------------\n"
          "\tTest Index\t|\tnonTOF\t|\tTOF\n"
          "------------------------------------------\n";
  for (int j = 0; j < num_tests; ++j)
    cerr << "\t\t" << std::to_string(j + 1) << "\t\t|\t"
         << ((test_results_nonTOF[j]) ? "Pass" : "Fail") << "\t|\t"
         << ((test_results_TOF[j]) ? "Pass" : "Fail") << std::endl;
}

void
ROOTConsistencyTests::
setup()
{
  // Create the point source (discretised_density_sptr) with GenerateImage from parameter file
  // GenerateImage requires `const char* const par_filename`.
  GenerateImage image_gen_application(get_generate_image_par_filename().c_str());
  image_gen_application.compute();
  discretised_density_sptr = image_gen_application.get_output_sptr();

  // Needs to be cast to VoxelsOnCartesianGrid to be able to calculate the centre of gravity
  const VoxelsOnCartesianGrid<float> &discretised_cartesian_grid =
      dynamic_cast<VoxelsOnCartesianGrid<float> &>(*discretised_density_sptr);

  // Find the center of mass and grid spacing of the original data
  original_coords = find_centre_of_gravity_in_mm(discretised_cartesian_grid);
  grid_spacing = discretised_cartesian_grid.get_grid_spacing();

  // For post_processing and debugging, create a list of the closest LOR coords
  nonTOF_closest_voxels_list.clear();
  TOF_LOR_peak_value_coords.clear();

  // Reset the number of processed and failed events
  num_failed_nonTOF_lor_events = 0;
  num_failed_TOF_lor_events = 0;
  num_events_tested = 0;

  // Failure conditioner and recording
  nonTOF_distance_threshold = 1.5 * norm(grid_spacing); // Using norm(grid_spacing) as a nonTOF_distance_threshold
  TOF_distance_threshold = 2.5 * norm(grid_spacing); // Using norm(grid_spacing) as a tof_distance_threshold
}

void
ROOTConsistencyTests::process_list_data()
{
  // Configure the list mode reader
  shared_ptr<CListModeData> lm_data_sptr(read_from_file<CListModeData>(get_root_header_filename()));
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
  proj_matrix_sptr.get()->set_up(lm_data_sptr->get_proj_data_info_sptr(),
                                 discretised_density_sptr);

  // loop over all events in the listmode file
  shared_ptr<CListRecord> record_sptr = lm_data_sptr->get_empty_record_sptr();
  CListRecord& record = *record_sptr;
  ProjMatrixElemsForOneBin proj_matrix_row;
  while (lm_data_sptr->get_next_record(record) == Succeeded::yes)
  {
    // only stores prompts
    if (record.is_event() && record.event().is_prompt())
      {
        Bin bin;
        bin.set_bin_value(1.f);
        // gets the bin corresponding to the event
        record.event().get_bin(bin, *lm_data_sptr->get_proj_data_info_sptr());
        if (bin.get_bin_value() > 0)
          {
            // computes the non-TOF probabilities along the bin LOR
            proj_matrix_sptr->get_proj_matrix_elems_for_one_bin(proj_matrix_row, bin);

            // Test event for non-TOF consistency
            if (!test_nonTOF_LOR_closest_approach(proj_matrix_row))
              num_failed_nonTOF_lor_events += 1;

            // Test TOF aspects of the LOR
            if (!test_TOF_max_lor_voxel(proj_matrix_row))
              num_failed_TOF_lor_events += 1;

            // Record as tested event
            num_events_tested += 1;
          }
      }
  }
}


//// NON-TOF TESTING METHODS ////
bool
ROOTConsistencyTests::test_nonTOF_LOR_closest_approach(const ProjMatrixElemsForOneBin& probabilities)
{
  // Loop variables
  CartesianCoordinate3D<float> closest_LOR_voxel_to_origin;
  float min_distance;  // shortest distance to the origin along the LOR
  bool first_entry = true; // Use this to populate with the initial value

  ProjMatrixElemsForOneBin::const_iterator element_ptr = probabilities.begin();
  // iterate over all to element_ptr to find the minimal distance between LOR and original_coords
  while (element_ptr != probabilities.end())
  {
    CartesianCoordinate3D<float> voxel_coords =
        discretised_density_sptr->get_physical_coordinates_for_indices(element_ptr->get_coords());

    float dist_to_original = norm(voxel_coords - original_coords);
    if (dist_to_original < min_distance || first_entry)
    {
      closest_LOR_voxel_to_origin = voxel_coords;
      min_distance = dist_to_original;
      first_entry = false;
    }
    ++element_ptr;
  }

  // Log closest voxel to origin in vector (value doesn't matter)
  nonTOF_closest_voxels_list.push_back(closest_LOR_voxel_to_origin);

  if (min_distance > nonTOF_distance_threshold)
    return false; // Test failed - LOR closest voxel beyond nonTOF_distance_threshold
  return true; // Test passed - LOR closest voxel within nonTOF_distance_threshold
}


void
ROOTConsistencyTests::
post_processing_nonTOF()
{
  cerr << "\nNon-TOF number of failed events: " << num_failed_nonTOF_lor_events
       << "\tNumber of tested events: = " << num_events_tested << std::endl;

  test_results_nonTOF[test_index-1] = check_if_less(num_failed_nonTOF_lor_events, failure_tolerance_nonTOF * num_events_tested,
                                                  "The number of failed TOF events is more than the tolerance(" +
                                                      std::to_string(100*failure_tolerance_nonTOF)+ "%)");

  { // Save the closest coordinate for each LOR to file.
    std::string lor_pos_filename = "non_TOF_voxel_data_" + std::to_string(test_index) + ".txt";
    cerr << "Saving debug information as: " << lor_pos_filename <<
            "\nThe first entry is the original coordinate position." << std::endl;
    std::ofstream myfile;
    myfile.open(lor_pos_filename.c_str());
    // The first entry is the original coords
    myfile << original_coords.x() << " " << original_coords.y() << " " << original_coords.z() << std::endl;
    for (std::vector<CartesianCoordinate3D<float>>::iterator coord_entry = nonTOF_closest_voxels_list.begin();
         coord_entry != nonTOF_closest_voxels_list.end(); ++coord_entry)
      myfile << coord_entry->x() << " " << coord_entry->y() << " " << coord_entry->z() << std::endl;
    myfile.close();
  }
}


//// TOF TESTING METHODS ////
bool
ROOTConsistencyTests::
test_TOF_max_lor_voxel(const ProjMatrixElemsForOneBin& probabilities)
{
  // Along a TOF LOR (probabilities), values assigned to voxels (or positions used here).
  // This method finds highest value voxel(s).
  // There may be more than one voxel with the highest value.
  // Everytime a new highest voxel value is found, sum_weighted_position is re-set to that value.
  // If a voxel is found with the same value as in sum_weighted_position, the coordinates are added.
  // At the end of the loop, divide the coordinates by `num_max_value_elements` and add to the list of max_value_voxels, 1 value per event.

  // Loop variables
  ProjMatrixElemsForOneBin::const_iterator element_ptr = probabilities.begin();
  int num_max_value_elements = 0;
  WeightedCoordinate sum_weighted_position(CartesianCoordinate3D<float>(0, 0, 0), 0);

  while (element_ptr != probabilities.end())
  {
    if (element_ptr->get_value() > sum_weighted_position.value)
    {
      // New highest value found, set sum_weighted_position to new value
      sum_weighted_position.coord = discretised_density_sptr->get_physical_coordinates_for_indices(element_ptr->get_coords());
      sum_weighted_position.value = element_ptr->get_value();
      num_max_value_elements = 1;
    }
    else if (element_ptr->get_value() == sum_weighted_position.value)
    {
      // Same value found, add coordinates to sum_weighted_position (will divide later)
      sum_weighted_position.coord += discretised_density_sptr->get_physical_coordinates_for_indices(element_ptr->get_coords());
      num_max_value_elements++;
    }
    ++element_ptr;
  }
  // Divide sum_weighted_position by num_max_value_elements to get the COM position
  sum_weighted_position.coord /= num_max_value_elements;

  // Add the COM position to the list of max_value_voxels
  TOF_LOR_peak_value_coords.push_back(sum_weighted_position);

  // Check if the COM position is within the TOF_distance_threshold tolerance
  if (norm(original_coords - sum_weighted_position.coord) > TOF_distance_threshold)
    return false; // Test failed - LOR closest voxel beyond tof_distance_threshold
  return true; // Test passed - LOR closest voxel within tof_distance_threshold
}


void
ROOTConsistencyTests::
post_processing_TOF()
{
  cerr << "\nTOF number of failed events: " << num_failed_TOF_lor_events
       << "\tNumber of tested events: = " << num_events_tested << std::endl;

  test_results_TOF[test_index-1] = check_if_less(num_failed_TOF_lor_events, failure_tolerance_TOF * num_events_tested,
                                               "The number of failed TOF events is more than the tolerance(" +
                                                   std::to_string(100*failure_tolerance_nonTOF)+ "%)");

  { // Save the closest coordinate for each LOR to file.
    std::string lor_pos_filename = "TOF_voxel_data_" + std::to_string(test_index) + ".txt";
    cerr << "Saving debug information as: " << lor_pos_filename <<
            "\nThe first entry is the original coordinate position." << std::endl;

    std::ofstream myfile;
    myfile.open(lor_pos_filename.c_str());
    myfile << original_coords.x() << " " << original_coords.y() << " " << original_coords.z() << std::endl;
    for (std::vector<WeightedCoordinate>::iterator coord_entry = TOF_LOR_peak_value_coords.begin();
         coord_entry != TOF_LOR_peak_value_coords.end(); ++coord_entry)
      myfile << coord_entry->coord.x() << " " << coord_entry->coord.y() << " " << coord_entry->coord.z() << std::endl;
    myfile.close();
  }
}

END_NAMESPACE_STIR

int main(int argc, char **argv)
{
  USING_NAMESPACE_STIR
  // Should be called from `STIR/examples/ROOT_files/ROOT_STIR_consistency`

  // Tests in class ROOTConsistencyTests
  ROOTConsistencyTests test;
  test.run_tests();
  return test.main_return_value();
}
