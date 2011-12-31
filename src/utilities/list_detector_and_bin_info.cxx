//
// $Id$
//
/*
    Copyright (C) 2011- $Date$, Hammersmith Imanet Ltd
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
  \file
  \ingroup utilities

  \brief A utility that lists bin and detector info for a given (cylindrical) scanner on stdout.

  \par Usage

  \code
  list_detector_and_bin_info scanner_name crystal1 crystal2 ring1 ring2
  \endcode

  This will list various things such as detection coordinates, bins etc to stdout.
  This is really only useful for developers who need to get their head round the
  STIR conventions.

  Currently bin info is listed for non-arccorrected projection data without any 
  compression.

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/LORCoordinates.h"
#include "stir/Bin.h"
#include "stir/Scanner.h"
#include "stir/stream.h"
#include <iostream> 

USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{ 
  
  if(argc!=6) 
  {
    std::cerr<<"Usage: " << argv[0] << " scanner_name crystal1 crystal2 ring1 ring2\n";
    std::cerr<<"\nLists detection and bin info for non-arccorrected projection data without any compression.\n";
    return EXIT_FAILURE;

  }
  shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name(argv[1]));
  if (scanner_sptr->get_type() == Scanner::Unknown_scanner)
    {
      std::cerr << "I did not recognise the scanner\n";
      return (EXIT_FAILURE);
    }

  shared_ptr<ProjDataInfoCylindricalNoArcCorr> proj_data_info_sptr
    (dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>
     (
     ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
                                   1, scanner_sptr->get_num_rings()-1,
                                   scanner_sptr->get_num_detectors_per_ring()/2,
                                   scanner_sptr->get_default_num_arccorrected_bins(), 
                                   false)
      ));



  {
    using std::cout;

    Bin bin;
    int det_num_a = atoi(argv[2]);
    int det_num_b = atoi(argv[3]);
    int ring_a = atoi(argv[4]);
    int ring_b = atoi(argv[5]);

    DetectionPositionPair<> det_pos;
    LORInAxialAndNoArcCorrSinogramCoordinates<float> lor;

    proj_data_info_sptr->get_bin_for_det_pair (bin,
                                               det_num_a, ring_a, det_num_b, ring_b);
    cout << "bin: (segment " << bin.segment_num() << ", axial pos " << bin.axial_pos_num()
         << ", view = " << bin.view_num() 
         << ", tangential_pos_num = " << bin.tangential_pos_num() << ")\n";
    cout << "bin coordinates: (tantheta: " << proj_data_info_sptr->get_tantheta(bin)
         << ", m: " << proj_data_info_sptr->get_m(bin)
         << ", phi: " << proj_data_info_sptr->get_phi(bin)
         << ", s: " << proj_data_info_sptr->get_s(bin)
         << ")\n";
    proj_data_info_sptr->get_LOR(lor, bin);
    cout << "LOR cylindrical: (z1: " << lor.z1() << ", z2: " << lor.z2()
         << ", phi: " << lor.phi() << ", beta: " << lor.beta() << " (= s: " << lor.s() << ")"
          << ")\n";

    LORAs2Points<float> lor_points;
    lor.get_intersections_with_cylinder(lor_points, scanner_sptr->get_effective_ring_radius());
    cout << "Detection position Cartesian: " << lor_points.p1() << lor_points.p2() <<'\n';

    proj_data_info_sptr->get_det_pos_pair_for_bin(det_pos, bin);
    cout << "Detection position index "
         <<"(c:" << det_pos.pos1().tangential_coord()
         << ",r:" << det_pos.pos1().axial_coord()
         << ",l:" << det_pos.pos1().radial_coord()
         << ")-"
         << "(c:" << det_pos.pos2().tangential_coord()
         << ",r:" << det_pos.pos2().axial_coord()
         << ",l:" << det_pos.pos2().radial_coord()
         << ")";

#if 0
    {
      CartesianCoordinate3D<float> coord_1, coord_2;
      proj_data_info_sptr->find_cartesian_coordinates_given_scanner_coordinates (coord_1,coord_2,
                                                                                 ring_a,ring_b, 
                                                                                 det_num_a, det_num_b);
      const CartesianCoordinate3D<float> 
        shift(scanner_sptr->get_ring_spacing()*(scanner_sptr->get_num_rings()-1)/2.F, 0.F,0.F);
      cout << "\nObsolete function:\n"
           << "find_cartesian_coordinates_given_scanner_coordinates (after shift in z)\n" 
           << coord_1 - shift << coord_2 - shift <<'\n';
      // will give same result as above
      proj_data_info_sptr->find_bin_given_cartesian_coordinates_of_detection(bin, coord_1, coord_2);
      cout << "bin: (segment " << bin.segment_num() << ", axial pos " << bin.axial_pos_num()
           << ", view = " << bin.view_num() 
           << ", tangential_pos_num = " << bin.tangential_pos_num() << ")\n";
    }
#endif

    cout << '\n';
  }

  return EXIT_SUCCESS;
}
