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
  \brief  Definition of CListEventDataECAT962, listmode event data for the ECAT 962 (aka Exact HR+)
    
  \author Nikos Efthimiou
  \author Kris Thielemans
      
*/

#ifndef __stir_listmode_CListEventDataECAT962_H__
#define __stir_listmode_CListEventDataECAT962_H__

#include "stir/IO/stir_ecat_common.h" // for namespace macros
#include "stir/Succeeded.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

/*!
  \class
  \ingroup listmode
  \brief  Class for storing and using a coincidence event from a listmode file

    The private definition is specific to the 962. Public members are generic
    though.

  For the 962 the event word is 32 bit. To save 1 bit in size, a 2d sinogram
     encoding is used (as opposed to a detector number on the ring
     for both events).
     Both bin and view use 9 bits, so their maximum range is
     512 values, which is fine for the 962 (which needs only 288).

   The 962 has 2 other bits, one for the energy window, and a 'multiple' bit.

  \todo use DetectionPosition etc.

  \author Kris Thielemans

*/

class CListEventDataECAT962 
{
 public:  
  inline bool is_prompt() const { return random == 0; }
  inline Succeeded set_prompt(const bool prompt = true) 
  { if (prompt) random=0; else random=1; return Succeeded::yes; }

/*! This routine returns the corresponding tangential_pos_num,view_num,ring_a and ring_b
   */
  void get_sinogram_and_ring_coordinates(int& view, int& tangential_pos_num, unsigned int& ring_a, unsigned int& ring_b) const;
  
/*! This routine constructs a coincidence event */
  void set_sinogram_and_ring_coordinates(
			const int view_num, const int tangential_pos_num, 
			const unsigned int ring_a, const unsigned int ring_b);


 private:
    /* ring encoding. use as follows:
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
  unsigned    block_A_ring_bit1 : 1;
  unsigned    block_B_ring_bit1 : 1;
  unsigned    block_A_ring_bit0 : 1;
  unsigned    block_B_ring_bit0 : 1;
  unsigned    block_B_detector : 3;
  unsigned    block_A_detector : 3;
  unsigned    scatter  : 1;
  unsigned    random  : 1;
  unsigned    multiple  : 1;
  unsigned    bin : 9;
  unsigned    view : 9;
#else
  // Do byteswapping first before using this bit field.
  unsigned    view : 9;
  unsigned    bin : 9;
  unsigned    multiple  : 1;
  unsigned    random  : 1;
  unsigned    scatter  : 1;
  unsigned    block_A_detector : 3;
  unsigned    block_B_detector : 3;
  unsigned    block_B_ring_bit0 : 1;
  unsigned    block_A_ring_bit0 : 1;
  unsigned    block_B_ring_bit1 : 1;
  unsigned    block_A_ring_bit1 : 1;
  unsigned    type    : 1; /* 0-coincidence event, 1-time tick */

#endif
}; /*-coincidence event*/ 

#if 0
//! Class for storing and using a coincidence event from a listmode file from the ECAT 962 scanner
class CListEventECAT962 : public CListEventCylindricalScannerWithViewTangRingRingEncoding<CListEventDataECAT962>
{
 public:
  CListEventECAT962() :
    CListEventCylindricalScannerWithViewTangRingRingEncoding<CListEventDataECAT962>(new Scanner(Scanner::E962))
    {}

  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  inline bool is_prompt() const { return this->data.random == 0; }
  inline Succeeded set_prompt(const bool prompt = true)
  { if (prompt) this->data.random=0; else this->data.random=1; return Succeeded::yes; }

 private:
  BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT962)==4);
  union
  {
    CListEventDataECAT962   data;
    boost::int32_t         raw;
  };
};
#endif

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
