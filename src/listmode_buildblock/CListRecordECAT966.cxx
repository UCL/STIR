//
// $Id$
//
/*
    Copyright (C) 1998- $Date$, Hammersmith Imanet Ltd
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
  \ingroup listmode
  \brief Implementation of classes CListEventECAT966 and CListRecordECAT966 
  for listmode events for the ECAT 966 (aka Exact 3d).
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/

#include "stir/listmode/CListRecordECAT966.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"

#include <algorithm>
#ifndef STIR_NO_NAMESPACES
using std::swap;
using std::streamsize;
using std::streampos;
#endif

START_NAMESPACE_STIR

// static members

shared_ptr<Scanner> 
CListRecordECAT966::
scanner_sptr =
  new Scanner(Scanner::E966);

shared_ptr<ProjDataInfoCylindricalNoArcCorr>
CListRecordECAT966::
uncompressed_proj_data_info_sptr =
   dynamic_cast<ProjDataInfoCylindricalNoArcCorr *>(
   ProjDataInfo::ProjDataInfoCTI(scanner_sptr, 
				 1, scanner_sptr->get_num_rings()-1,
				 scanner_sptr->get_num_detectors_per_ring()/2,
				 scanner_sptr->get_default_num_arccorrected_bins(), 
				 false));


/*	Global Definitions */
static const int  MAXPROJBIN = 512;
/* data for the 966 scanner */
static const int CRYSTALRINGSPERDETECTOR = 8;

void
CListEventDataECAT966::
get_sinogram_and_ring_coordinates(
		   int& view_num, int& tangential_pos_num, int& ring_a, int& ring_b) const
{
  const int NumProjBins = MAXPROJBIN;
  const int NumProjBinsBy2 = MAXPROJBIN / 2;

  view_num = view;
  tangential_pos_num = bin;
  /* KT 31/05/98 use >= in comparison now */
  if ( tangential_pos_num >= NumProjBinsBy2 )
      tangential_pos_num -= NumProjBins ;

  ring_a = ( block_A_ring * CRYSTALRINGSPERDETECTOR ) +  block_A_detector ;
  ring_b = ( block_B_ring * CRYSTALRINGSPERDETECTOR ) +  block_B_detector ;
}

void 
CListEventDataECAT966::
set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const int ring_a, const int ring_b)
{
  const int NumProjBins = MAXPROJBIN;
  type = 0;
  block_A_ring     = ring_a / CRYSTALRINGSPERDETECTOR;
  block_A_detector = ring_a % CRYSTALRINGSPERDETECTOR;
  block_B_ring     = ring_b / CRYSTALRINGSPERDETECTOR;
  block_B_detector = ring_b % CRYSTALRINGSPERDETECTOR;

  bin = tangential_pos_num < 0 ? tangential_pos_num + NumProjBins : tangential_pos_num;
  view = view_num;
}


void 
CListEventDataECAT966::
get_detectors(
		   int& det_num_a, int& det_num_b, int& ring_a, int& ring_b) const
{
  int tangential_pos_num;
  int view_num;
  get_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);

  CListRecordECAT966::
    get_uncompressed_proj_data_info_sptr()->
    get_det_num_pair_for_view_tangential_pos_num(det_num_a, det_num_b, 
						 view_num, tangential_pos_num);
}

void 
CListEventDataECAT966::
set_detectors(
			const int det_num_a, const int det_num_b,
			const int ring_a, const int ring_b)
{
  int tangential_pos_num;
  int view_num;
  const bool swap_rings =
  CListRecordECAT966::
    get_uncompressed_proj_data_info_sptr()->
    get_view_tangential_pos_num_for_det_num_pair(view_num, tangential_pos_num,
						 det_num_a, det_num_b);

  if (swap_rings)
  {
    set_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);
  }
  else
  {
     set_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_b, ring_a);
  }
}

// TODO maybe move to ProjDataInfoCylindricalNoArcCorr
static void
sinogram_coordinates_to_bin(Bin& bin, const int view_num, const int tang_pos_num, 
			const int ring_a, const int ring_b,
			const ProjDataInfoCylindrical& proj_data_info)
{
  if (proj_data_info.get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), ring_a, ring_b) ==
      Succeeded::no)
    {
      bin.set_bin_value(-1);
      return;
    }
  bin.set_bin_value(1);
  bin.view_num() = view_num / proj_data_info.get_view_mashing_factor();  
  bin.tangential_pos_num() = tang_pos_num;
}

void 
CListRecordECAT966::
get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const
{
  assert (dynamic_cast<const ProjDataInfoCylindricalNoArcCorr *>(&proj_data_info)!=0);

  int tangential_pos_num;
  int view_num;
  int ring_a;
  int ring_b;
  event_data.get_sinogram_and_ring_coordinates(view_num, tangential_pos_num, ring_a, ring_b);
  sinogram_coordinates_to_bin(bin, view_num, tangential_pos_num, ring_a, ring_b, 
			      static_cast<const ProjDataInfoCylindrical&>(proj_data_info));
}

void
CListRecordECAT966::
get_detection_coordinates(CartesianCoordinate3D<float>& coord_1,
			  CartesianCoordinate3D<float>& coord_2) const
{
  int det_num_a, det_num_b, ring_a, ring_b;
  event_data.get_detectors(det_num_a, det_num_b, ring_a, ring_b);

  uncompressed_proj_data_info_sptr->
    find_cartesian_coordinates_given_scanner_coordinates(coord_1, coord_2,
							 ring_a, ring_b,
							 det_num_a, det_num_b);
}

void 
CListRecordECAT966::
get_uncompressed_bin(Bin& bin) const
{
  int ring_a;
  int ring_b;
  event_data.get_sinogram_and_ring_coordinates(bin.view_num(), bin.tangential_pos_num(), ring_a, ring_b);
  uncompressed_proj_data_info_sptr->
    get_segment_axial_pos_num_for_ring_pair(bin.segment_num(), bin.axial_pos_num(), 
					    ring_a, ring_b);
}  



END_NAMESPACE_STIR
