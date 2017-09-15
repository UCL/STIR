/*
    Copyright (C) 2017, UCL
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
  Implementation of stir::test_consistency_root
*/
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/ProjMatrixElemsForOneBin.h"
#include "stir/centre_of_gravity.h"
#include "stir/listmode/LmToProjData.h"
#include "stir/listmode/CListModeDataROOT.h"
#include "stir/listmode/CListRecord.h"
#include "stir/IO/read_from_file.h"
#include "stir/HighResWallClockTimer.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/shared_ptr.h"
#include "stir/RunTests.h"
#include "boost/lexical_cast.hpp"

#include "stir/info.h"
#include "stir/warning.h"

#include <stack>

using std::cerr;
using std::ifstream;

START_NAMESPACE_STIR

/*!
 *  \ingroup recon_test
 *  \brief Test class to check the consistency between ROOT listmode and STIR backprojection for high resolution TOF
 *  \author Elise Emond
 *
 * This test currently uses Root listmodes of single point sources. Scatters are not
 * considered. It could be extended to actual scanner data, for which we would need
 * to exclude scatter events. A way to do so would be to compute the distance between
 * the bin LOR and the (non-TOF?) LOR calculated from the original data and exclude
 * the events for which the distance would be more than a chosen threshold.
 *
 */

class ROOTconsistency_Tests : public RunTests
{
public:
  ROOTconsistency_Tests(std::string in, std::string image)
    : root_header_filename(in), image_filename(image)
    {}
    void run_tests();

    // Class to store the coordinates and weights of the maxima of the Lines-of-Response
    // used to calculate the centre of gravity (see below).
    class LORMax{
    public:
        LORMax() { voxel_centre = CartesianCoordinate3D<float>(0.f,0.f,0.f); value = 0.f;}
        CartesianCoordinate3D<float> voxel_centre;
        float value;
    };

private:
    //! Reads listmode event by event, computes the ProjMatrixElemsForOneBin (probabilities
    //! along a bin LOR) and stores in a vector the coordinates and weights of the
    //! LOR maxima (vector::LORMax) prior to computing the centre of mass of those.
	void construct_list_of_LOR_max(
	    const shared_ptr<DiscretisedDensity<3, float> >& test_discretised_density_sptr);
	
	//! Selects and stores the highest probability elements of ProjMatrixElemsForOneBin.
	void get_LOR_of_max(const ProjMatrixElemsForOneBin& probabilities,
	    const shared_ptr<DiscretisedDensity<3, float> >& test_discretised_density_sptr);

	//! Given a vector::LORMax, computes the centre of mass.
	CartesianCoordinate3D<float> compute_centre_of_mass();

	//! Checks if original and calculated coordinates are close enough.
	void compare_original_and_calculated_coordinates(
	    const CartesianCoordinate3D<float>& original_coords,
        const CartesianCoordinate3D<float>& centre_of_mass,
        const BasicCoordinate<3, float>& grid_spacing);

	//! Modified version of check_if_equal for this test
	bool check_if_almost_equal(const double a, const double b, std::string str, const double tolerance);

	std::string root_header_filename;
	std::string image_filename;
    std::vector<LORMax> max_lor;
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
    CartesianCoordinate3D<float> original_coords = find_centre_of_gravity_in_mm(discretised_cartesian_grid);

    construct_list_of_LOR_max(discretised_density_sptr);

    CartesianCoordinate3D<float> centreofmass = compute_centre_of_mass();

    compare_original_and_calculated_coordinates(original_coords,centreofmass,discretised_cartesian_grid.get_grid_spacing());
}

void
ROOTconsistency_Tests::
construct_list_of_LOR_max(const shared_ptr<DiscretisedDensity<3, float> >& discretised_density_sptr)
{
	shared_ptr<CListModeData> lm_data_sptr(read_from_file<CListModeData>(root_header_filename));

	shared_ptr<ProjMatrixByBin> proj_matrix_sptr(new ProjMatrixByBinUsingRayTracing());

	proj_matrix_sptr.get()->set_up(lm_data_sptr->get_proj_data_info_sptr(),
	        discretised_density_sptr);
    proj_matrix_sptr->enable_tof(lm_data_sptr->get_proj_data_info_sptr());

    ProjMatrixElemsForOneBin proj_matrix_row;

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
                  get_LOR_of_max(proj_matrix_row, discretised_density_sptr);
                }
	        }
	    }
	  }

}

void
ROOTconsistency_Tests::get_LOR_of_max(const ProjMatrixElemsForOneBin& probabilities, const shared_ptr<DiscretisedDensity<3,
    float> >& test_discretised_density_sptr)
{
  std::stack<LORMax> tmp_max_lor;

  CartesianCoordinate3D<float> voxel_centre;

  float maxLOR = 0;

  ProjMatrixElemsForOneBin::const_iterator element_ptr = probabilities.begin();
  // iterative calculation of highest probability and corresponding elements along the LOR
  while (element_ptr != probabilities.end())
  {
    if (element_ptr->get_value() >= maxLOR)
    {
      maxLOR = element_ptr->get_value();
      voxel_centre =
              test_discretised_density_sptr->get_physical_coordinates_for_indices(element_ptr->get_coords());
      LORMax tmp;
      tmp.value = element_ptr->get_value();
      tmp.voxel_centre = voxel_centre;
      tmp_max_lor.push(tmp);
    }
    ++element_ptr;
  }

  // only selects the elements on top of the stack, corresponding to the highest probability
  if (maxLOR !=0)
  {
    while(!tmp_max_lor.empty())
    {
      if (tmp_max_lor.top().value == maxLOR)
      {
        max_lor.push_back(tmp_max_lor.top());
        tmp_max_lor.pop();
      }
      else break;
    }
  }
}

CartesianCoordinate3D<float> ROOTconsistency_Tests::compute_centre_of_mass()
{

  // creation of a file with all LOR maxima, to be able to plot them
  std::ofstream myfile;
  std::string file_name = image_filename.substr(0,image_filename.size()-3) + ".txt";
  myfile.open (file_name.c_str());

  LORMax centreofmass;

  // computes centre of mass
  for (std::vector<LORMax>::iterator lor_element_ptr=max_lor.begin();
          lor_element_ptr != max_lor.end();++lor_element_ptr)
  {
    centreofmass.voxel_centre.x() += lor_element_ptr->voxel_centre.x()*lor_element_ptr->value;
    centreofmass.voxel_centre.y() += lor_element_ptr->voxel_centre.y()*lor_element_ptr->value;
    centreofmass.voxel_centre.z() += lor_element_ptr->voxel_centre.z()*lor_element_ptr->value;
    centreofmass.value += lor_element_ptr->value;

    myfile << lor_element_ptr->voxel_centre.x() << " "
           << lor_element_ptr->voxel_centre.y() << " "
           << lor_element_ptr->voxel_centre.z() << " "
           << lor_element_ptr->value << std::endl;

  }

  // needs to divide by the weights
  if (centreofmass.value != 0)
  {
    centreofmass.voxel_centre.x()=centreofmass.voxel_centre.x()/centreofmass.value;
    centreofmass.voxel_centre.y()=centreofmass.voxel_centre.y()/centreofmass.value;
    centreofmass.voxel_centre.z()=centreofmass.voxel_centre.z()/centreofmass.value;
  }
  else
  {
    warning("Total weight of the centre of mass equal to 0. Please check your data.");
    centreofmass.voxel_centre.x()=0;
    centreofmass.voxel_centre.y()=0;
    centreofmass.voxel_centre.z()=0;
  }

  cerr << "Centre of gravity coordinates: " << centreofmass.voxel_centre.x() << " "
      << centreofmass.voxel_centre.y() << " " << centreofmass.voxel_centre.z() << std::endl;

  myfile.close();

  return centreofmass.voxel_centre;

}

// TODO change this
void ROOTconsistency_Tests::compare_original_and_calculated_coordinates(const CartesianCoordinate3D<float>& original_coords,
    const CartesianCoordinate3D<float>& centre_of_mass, const BasicCoordinate<3, float>& grid_spacing)
{
  check_if_almost_equal(static_cast<double>(original_coords.x()),static_cast<double>(centre_of_mass.x()),"x",grid_spacing[1]);
  check_if_almost_equal(static_cast<double>(original_coords.y()),static_cast<double>(centre_of_mass.y()),"y",grid_spacing[2]);
  check_if_almost_equal(static_cast<double>(original_coords.z()),static_cast<double>(centre_of_mass.z()),"z",grid_spacing[3]);

  cerr << "Original coordinates: " << original_coords.x() << " "
      << original_coords.y() << " " << original_coords.z() << std::endl;
}

bool
ROOTconsistency_Tests::check_if_almost_equal(const double a, const double b, std::string str, const double tolerance)
{
  if ((fabs(a-b) > tolerance))
  {
    std::cerr << "Error : unequal values are " << a << " and " << b
         << ". " << str << std::endl;
    everything_ok = false;
    return false;
  }
  else
    return true;
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
