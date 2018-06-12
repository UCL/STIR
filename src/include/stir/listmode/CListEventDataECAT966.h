//
//
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
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
  \brief Classes for listmode events for the ECAT 966 (aka Exact 3d)
    
  \author Nikos Efthimiou  
  \author Kris Thielemans
      
*/

#ifndef __stir_listmode_CListEventDataECAT966_H__
#define __stir_listmode_CListEventDataECAT966_H__

#include "stir/IO/stir_ecat_common.h" // for namespace macros

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

//! Class for decoding storing and using a raw coincidence event from a listmode file from the ECAT 966 scanner
/*! \ingroup listmode

     This class just provides the bit-field definitions. You should normally use CListEventECAT966.

     For the 966 the event word is 32 bit. To save 1 bit in size, a 2d sinogram
     encoding is used (as opposed to a detector number on the ring
     for both events).
     Both bin and view use 9 bits, so their maximum range is
     512 values, which is fine for the 966 (which needs only 288).

*/
class CListEventDataECAT966
{
 public:
  
  /*! This routine returns the corresponding tangential_pos_num,view_num,ring_a and ring_b
   */
  void get_sinogram_and_ring_coordinates(int& view, int& tangential_pos_num, unsigned int& ring_a, unsigned int& ring_b) const;
  
  /*! This routine constructs a coincidence event */
  void set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const int ring_a, const int ring_b);

  /* ring encoding. use as follows:
       #define CRYSTALRINGSPERDETECTOR 8
       ringA = ( block_A_ring * CRYSTALRINGSPERDETECTOR ) + block_A_detector ;
       in practice, for the 966 block_A_ring ranges from 0 to 5.

       This organisation corresponds to physical detector blocks (which
       have 8 crystal rings). Names are not very good probably...
       */				
    /* 'random' bit:
        1 if event is Random (it fell in delayed time window) */
    /* bin field  is shifted in a funny way, use the following code to find
       bin_number:
         if ( bin > NumProjBinsBy2 ) bin -= NumProjBins ;
	 */

#if STIRIsNativeByteOrderBigEndian
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */
  unsigned    block_B_ring : 3;
  unsigned    block_A_ring : 3;
  unsigned    block_B_detector : 3;
  unsigned    block_A_detector : 3;
  unsigned    random  : 1;
  unsigned    bin : 9;
  unsigned    view : 9;
#else
  // Do byteswapping first before using this bit field.
  unsigned    view : 9;
  unsigned    bin : 9;
  unsigned    random  : 1;
  unsigned    block_A_detector : 3;
  unsigned    block_B_detector : 3;
  unsigned    block_A_ring : 3;
  unsigned    block_B_ring : 3;
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */

#endif
}; /*-coincidence event*/

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
