/*!

  \file
  \ingroup utilities

  \brief Construct randoms as a product of singles estimates

  \author Kris Thielemans

*/
/*
  Copyright (C) 2001- 2012, Hammersmith Imanet Ltd
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

#include "stir/ML_norm.h"

#include "stir/ProjDataInterfile.h"
#include "stir/multiply_crystal_factors.h"
#include "stir/GeometryBlocksOnCylindrical.h"
#include "stir/DetectorCoordinateMap.h"
#include "stir/stream.h"
#include "stir/IndexRange2D.h"
#include <iostream>
#include <fstream>
#include <string>
//#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ifstream;
using std::string;
#endif

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc!=5)
    {
      cerr << "Usage: " << argv[0] 
           << " out_filename in_norm_filename_prefix template_projdata eff_iter_num\n";
      return EXIT_FAILURE;
    }
  const int eff_iter_num = atoi(argv[4]);
  const int iter_num = 1;//atoi(argv[5]);
  //const bool apply_or_undo = atoi(argv[4])!=0;
  shared_ptr<ProjData> template_projdata_ptr = ProjData::read_from_file(argv[3]);
  const string in_filename_prefix = argv[2];
  const string output_file_name = argv[1];
  const string program_name = argv[0];

  ProjDataInterfile 
    proj_data(template_projdata_ptr->get_exam_info_sptr(),
              template_projdata_ptr->get_proj_data_info_sptr()->create_shared_clone(), 
	      output_file_name);

  const int num_rings = 
    template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_num_rings();
  const int num_detectors_per_ring = 
    template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_ptr()->get_num_detectors_per_ring();
  DetectorEfficiencies efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));

  {

    // efficiencies
      {
	char *in_filename = new char[in_filename_prefix.size() + 30];
	sprintf(in_filename, "%s_%s_%d_%d.out", 
		in_filename_prefix.c_str(), "eff", iter_num, eff_iter_num);
	ifstream in(in_filename);
	in >> efficiencies;
  // perform shifting of efficiency only if crystal map is defined in interfile header.
  if(template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_sptr()->get_detector_map_sptr() != nullptr){
    shared_ptr<const DetectorCoordinateMap> external_map = template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_sptr()->get_detector_map_sptr();
    shared_ptr<GeometryBlocksOnCylindrical> internal_map;
    internal_map.reset(new GeometryBlocksOnCylindrical(*(template_projdata_ptr->get_proj_data_info_sptr()->get_scanner_sptr())));
    DetectorEfficiencies internal_efficiencies(IndexRange2D(num_rings, num_detectors_per_ring));
    for (int axial = 0; axial <= num_rings; axial++)
    {
      for (int tang = 0; tang <= num_detectors_per_ring; tang++)
      {
        // (internal_axial, internal_tang, 0) <-> (x,y,z) <-> (external_axial, external_tang, 0)
        // therefore external map and internal map NEEDS TO AGREE on (x,y,z)
        stir::DetectionPosition<> internal_det_pos;
        internal_det_pos.radial_coord()=0;
        internal_det_pos.axial_coord()=axial;
        internal_det_pos.tangential_coord()=tang;
        stir::CartesianCoordinate3D<float> internal_3D_coord = internal_map->get_coordinate_for_det_pos(internal_det_pos);
        stir::DetectionPosition<> external_det_pos;
        if(external_map->find_detection_position_given_cartesian_coordinate(external_det_pos,internal_3D_coord) == Succeeded::yes){
          internal_efficiencies[axial, tang] = efficiencies[external_det_pos.axial_coord(), external_det_pos.tangential_coord()];
        }else{
          error("there is a mismatch of (x,y,z) coordinate between internal and external crystal. Check if depth of interaction between 2 crystal map matches!");
        }
      }
    }
    efficiencies = internal_efficiencies;
  }
	if (!in)
	  {
	    error("Error reading %s, using all 1s instead\n", in_filename);
	  }
	delete[] in_filename;
      }
  }

  multiply_crystal_factors(proj_data, efficiencies, 1.F);

  return EXIT_SUCCESS;
}
