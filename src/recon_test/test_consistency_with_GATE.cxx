/*
    Copyright (C) 2017, 2022, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license
    See STIR/LICENSE.txt for details
*/
/*!
 *  \ingroup recon_test
 *  \brief Test class to check the consistency between GATE and STIR
 *  \author Elise Emond
 *  \author Robert Twyman
 *
 * This test currently uses ROOT list mode data of single point sources to assess the consistency between GATE and STIR.
 * Each test aims to verify that each list mode event back projection is reasonably close to the point source position used
 * to generate the list mode data.
 * Currently there are 8 point sources, one in each of 8 list mode files, tested.
 *
 * This test is designed to run from ${STIR_SOURCE_PATH}/examples/ROOT_files/ROOT_STIR_consistency
 *
 * Non-TOF tests
 * This test computes the distance between the original point source position and the closes voxel that the list mode
 * event passes through. Ideally each event would travel directly through the original point source position but
 * error may be present. Therefore we test that the majority of LORs travel close enough.
 *
 * TOF tests
 * Follows approximately the same procedure as non-TOF tests but, instead of closest approach voxel along the LOR, the voxel
 * with maximum value is tested. If that voxel exceeds the threshold distance, the test is considered a failure.
 * Again, if the majority of LORs travel close enough, the test is considered a success.
 *
 * In both cases, the logs of voxel positions (origin and closest approach or maximum voxel intensity)
 * are written to files for later analysis.
 */

#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/centre_of_gravity.h"
#include "stir/listmode/CListModeDataROOT.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/read_from_file.h"
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


class GATEConsistencyTests : public RunTests
{
public:
  GATEConsistencyTests()= default;

  /*!
   * \brief Run the tests with ROOT data
   * Main method for running the tests.
   * For each list mode file, setups up the original point source positions and runs the tests and post processes the results.
   * Records the pass failure of each file and logs to console.
   */
  void run_tests() override;

private:

  /*! Initialise the original point source position by generating image and finding the centre of gravity
   * and sets TOF and non-TOF threshold distances and the number of events passed and failed counters
   */
  void setup();

  /*! Reads listmode event by event, computes the ProjMatrixElemsForOneBin (probabilities
   * along a bin LOR). This method should setup the proj_matrix_sptr.
   * The list data is processes once and each proj_matrix_row is passes to relevant non-TOF and TOF test methods.
   */
  void process_list_data();

  /*!
   * Performs the analysis steps for each test after the list mode data has been processed.
   */
  void post_processing();

  /*! Test if the closest voxel in the LOR (probabilities) to the original_coords
   * is within nonTOF_distance_threshold distance. If it is, pass with true, otherwise fales.
   * @param probabilities ProjMatrixElemsForOneBin object of a list mode event
   * @return True if test passes, false if failed.
   */
  bool test_nonTOF_LOR_closest_approach(const ProjMatrixElemsForOneBin& probabilities);

  /*! If enough of the non-TOF LORs pass the closest approach test, pass the test, otherwise fail.
   * Additionally, save the original and closest approach voxel positions to file.
   */
  void post_processing_nonTOF();


  /*! Test if the voxel with the highest value in the LOR (probabilities) is within TOF_distance_threshold to the original_coords.
   * If it is, pass with true, otherwise fales.
   * @param probabilities ProjMatrixElemsForOneBin object of a list mode event
   * @return \c true if within TOF_distance_threshold of original_coords, else \c false.
   */
  bool test_TOF_max_lor_voxel(const ProjMatrixElemsForOneBin& probabilities);

  /*! If enough of the TOF LORs pass the max voxel test distance test, pass, otherwise fail.
   * Additionally, save the original and max voxel positions of the LOR to file.
   */
  void post_processing_TOF();

  //! Logs the results of each test to console
  void log_results_to_console();

  // Auxiliary methods
  std::string get_root_header_filename() { return "pretest_output/root_header_test" + std::to_string(test_index) + ".hroot"; }
  std::string get_generate_image_par_filename() { return "SourceFiles/generate_image" + std::to_string(test_index) + ".par"; }


  ///// Class VARIABLES /////

private:
  // Data set variables
  int test_index = 0;
  int num_tests = 8;

  // Test results storage. Records if each test failed or not. E.g.`test_index` result is stored in `test_results_nonTOF[test_index -1]`
  std::vector<bool> test_results_nonTOF;
  std::vector<bool> test_results_TOF;

  shared_ptr<CListModeData> lm_data_sptr;

  // Original point source position variables
  shared_ptr<DiscretisedDensity<3, float> > discretised_density_sptr;
  CartesianCoordinate3D<float> original_coords; // Stored in class because of dynamic_cast
  CartesianCoordinate3D<float> grid_spacing;    // Stored in class because of dynamic_cast

  //! The number of events/LORs have been tested
  int num_events_tested = 0;

  /// NON TOF VARIABLES
  //! A vector to store the coordinates of the closest approach voxels.
  std::vector<CartesianCoordinate3D<float>> nonTOF_closest_voxels_list;
  //! Fraction of nonTOF LORs that can fail the closest approach test and data set still passes.
  const float failure_tolerance_nonTOF = 0.05;
  //! The number of nonTOF LORs that failed the closest approach test.
  int num_failed_nonTOF_lor_events = 0;
  //! The threshold distance for the closest approach test.
  double nonTOF_distance_threshold;

  /// TOF VARIABLES
  //! A vector to store the coordinates of the maximum voxels for each LOR.
  std::vector<WeightedCoordinate> TOF_LOR_peak_value_coords;
  //! Fraction of TOF LORs that can fail the maximum LOR intensity test and data set still passes.
  const float failure_tolerance_TOF = 0.05;
  //! The number of TOF LORs that failed the maximum LOR voxel intensity test.
  int num_failed_TOF_lor_events = 0;
  //! The threshold distance for the maximum LOR voxel intensity test.
  double TOF_distance_threshold;
};

void
GATEConsistencyTests::run_tests()
{
  test_results_nonTOF = std::vector<bool> (num_tests, true);
  test_results_TOF = std::vector<bool> (num_tests, true);

  cerr << "Testing consistency between GATE/ROOT and STIR. \n";
  for (int i = 1; i <= num_tests; ++i)
  {
    test_index = i;  // set the class variable to keep track of which test is being run
    cerr << "\nTesting dataset " << std::to_string(test_index) << "...\n";
    setup();
    process_list_data();
    post_processing();
  }
  log_results_to_console();
}

void
GATEConsistencyTests::
setup()
{
  // Initialise the list mode data object
  lm_data_sptr = read_from_file<CListModeData>(get_root_header_filename());

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
  {
    // With the default files, we found that 3.3*norm(grid_spacing) is a reasonable limit.
    // We try to generalise this to other data (although it will likely need further
    // adjustment for cases with very different TOF resolution).
    const float org_threshold = 3.3F / 75; // Through trial and error, this is about the minimum threshold

    const auto& pdi = *lm_data_sptr->get_proj_data_info_sptr();
    const float approx_tof_resolution = // 75 in initial configuration
        pdi.get_scanner_ptr()->get_timing_resolution() * pdi.get_tof_mash_factor();
    this->TOF_distance_threshold = approx_tof_resolution * org_threshold * norm(grid_spacing);
  }
}

void
GATEConsistencyTests::
process_list_data()
{
  // Configure the list mode reader
  shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());
  proj_matrix_sptr->set_up(lm_data_sptr->get_proj_data_info_sptr(),
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
GATEConsistencyTests::test_nonTOF_LOR_closest_approach(const ProjMatrixElemsForOneBin& probabilities)
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
GATEConsistencyTests::
post_processing_nonTOF()
{
  cerr << "\nNon-TOF number of failed events: " << num_failed_nonTOF_lor_events
       << "\tNumber of tested events: = " << num_events_tested << std::endl;

  test_results_nonTOF[test_index-1] = check_if_less(num_failed_nonTOF_lor_events, failure_tolerance_nonTOF * num_events_tested,
                                                  "The number of failed TOF events is more than the tolerance(" +
                                                      std::to_string(100*failure_tolerance_nonTOF)+ "%)");

  { // Save the closest coordinate for each LOR to file.
    std::string lor_pos_filename = "non_TOF_voxel_data_" + std::to_string(test_index) + ".csv";
    cerr << "Saving debug information as: " << lor_pos_filename << "\n"
         << "The first entry is the distance tolerance granted.\n"
         << "The second entry is the original coordinate position.\n";
    std::ofstream myfile;
    myfile.open(lor_pos_filename.c_str());
    // The first entry is the tolerance granted
    myfile << "tolerance," << nonTOF_distance_threshold << "\n";
    // The second entry is the original coords
    myfile << "original coordinates," << original_coords.x() << "," << original_coords.y() << "," << original_coords.z() << "\n";
    int i = 0;
    for (auto & entry : nonTOF_closest_voxels_list)
      {
        myfile << i << "," << entry.x() << "," << entry.y() << "," << entry.z() << "\n";
        i++;
      }
    myfile.close();
  }
}

void
GATEConsistencyTests::
post_processing()
{
  cerr << "\nResults for dataset: " << std::to_string(this->test_index) << std::endl;
  post_processing_nonTOF();
  post_processing_TOF();
};

//// TOF TESTING METHODS ////
bool
GATEConsistencyTests::
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
GATEConsistencyTests::
post_processing_TOF()
{
  cerr << "\nTOF number of failed events: " << num_failed_TOF_lor_events
       << "\tNumber of tested events: = " << num_events_tested << std::endl;

  test_results_TOF[test_index-1] = check_if_less(num_failed_TOF_lor_events, failure_tolerance_TOF * num_events_tested,
                                               "The number of failed TOF events is more than the tolerance(" +
                                                   std::to_string(100*failure_tolerance_nonTOF)+ "%)");

  { // Save the closest coordinate for each LOR to file.
    std::string lor_pos_filename = "TOF_voxel_data_" + std::to_string(test_index) + ".csv";
    cerr << "Saving debug information as: " << lor_pos_filename << "\n"
         << "The first entry is the distance tolerance granted.\n"
         << "The second entry is the original coordinate position.\n";

    std::ofstream myfile;
    myfile.open(lor_pos_filename.c_str());
    // The first entry is the tolerance granted
    myfile << "tolerance," << TOF_distance_threshold << "\n";
    // The second entry is the original coords
    myfile << "original coordinates," << original_coords.x() << "," << original_coords.y() << "," << original_coords.z() << "\n";
    int i = 0;
    for (auto & entry : TOF_LOR_peak_value_coords)
      {
        myfile << i << "," << entry.coord.x() << "," << entry.coord.y() << "," << entry.coord.z() << "\n";
        i++;
      }
    myfile.close();
  }
}

void
GATEConsistencyTests::
log_results_to_console()
{
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

END_NAMESPACE_STIR

int main(int argc, char **argv)
{
  USING_NAMESPACE_STIR
  // Should be called from `${STIR_SOURCE_PATH}/examples/ROOT_files/ROOT_STIR_consistency`

  // Tests in class GATEConsistencyTests
  GATEConsistencyTests test;
  test.run_tests();
  return test.main_return_value();
}
