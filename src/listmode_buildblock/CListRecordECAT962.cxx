//
//
/*
    Copyright (C) 1998- 2011, Hammersmith Imanet Ltd
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
  \brief Implementation of classes CListEventECAT962 and CListRecordECAT962
  for listmode events for the  ECAT 962 (aka Exact HR+).
    
  \author Kris Thielemans
      
*/

#include "stir/listmode/CListRecordECAT962.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7


/*	Global Definitions */
static const int  MAXPROJBIN = 512;
/* data for the 962 scanner */
static const int CRYSTALRINGSPERDETECTOR = 8;
//TODO NK check
void
CListEventDataECAT962::
get_sinogram_and_ring_coordinates(
		   int& view_num, int& tangential_pos_num, unsigned int& ring_a, unsigned int& ring_b) const
{
  const int NumProjBins = MAXPROJBIN;
  const int NumProjBinsBy2 = MAXPROJBIN / 2;

  view_num = view;
  tangential_pos_num = bin;
  /* KT 31/05/98 use >= in comparison now */
  if ( tangential_pos_num >= NumProjBinsBy2 )
      tangential_pos_num -= NumProjBins ;

  ring_a = ( (block_A_ring_bit0 + 2*block_A_ring_bit1) 
	     * CRYSTALRINGSPERDETECTOR ) +  block_A_detector ;
  ring_b = ( (block_B_ring_bit0 + 2*block_B_ring_bit1)
	     * CRYSTALRINGSPERDETECTOR ) +  block_B_detector ;
}

void 
CListEventDataECAT962::
set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const unsigned int ring_a, const unsigned int ring_b)
{
  const int NumProjBins = MAXPROJBIN;
  type = 0;
  const unsigned int block_A_ring     = ring_a / CRYSTALRINGSPERDETECTOR;
  block_A_detector = ring_a % CRYSTALRINGSPERDETECTOR;
  const unsigned int block_B_ring     = ring_b / CRYSTALRINGSPERDETECTOR;
  block_B_detector = ring_b % CRYSTALRINGSPERDETECTOR;

  assert(block_A_ring<4);
  block_A_ring_bit0 = block_A_ring | 0x1;
  block_A_ring_bit1 = block_A_ring/2;
  assert(block_B_ring<4);
  block_B_ring_bit0 = block_B_ring | 0x1;
  block_B_ring_bit1 = block_B_ring/2;
  
  bin = tangential_pos_num < 0 ? tangential_pos_num + NumProjBins : tangential_pos_num;
  view = view_num;
}


END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR
