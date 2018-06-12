/*
    Copyright (C) 2003-2011 Hammersmith Imanet Ltd
    Copyright (C) 2013-2014 University College London
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
  \brief Classes for listmode events for the ECAT 8 format
    
  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#ifndef __stir_listmode_CListEventECAT8_32bit_H__
#define __stir_listmode_CListEventECAT8_32bit_H__

#include "stir/listmode/CListEventDataECAT8_32bit.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"


START_NAMESPACE_STIR
namespace ecat {
    

/*!
 * \class
 * \brief Class for storing and using a coincidence event from a listmode file from Siemens scanners using the ECAT 8_32bit format
 * \todo This implementation only works if the list-mode data is stored without axial compression.
  \todo If the target sinogram has the same characteristics as the sinogram encoding used in the list file 
  (via the offset), the code could be sped-up dramatically by using the information. 
  At present, we go a huge round-about (offset->sinogram->detectors->sinogram->offset)

  \author Kris Thielemans
*/
class CListEventECAT8_32bit : public CListEventCylindricalScannerWithDiscreteDetectors
{

 public:
  typedef CListEventDataECAT8_32bit DataType;
  DataType get_data() const { return this->data; }

 public:  
  CListEventECAT8_32bit(const shared_ptr<ProjDataInfo>& proj_data_info_sptr);

 //! This routine returns the corresponding detector pair   
  virtual void get_detection_position(DetectionPositionPair<>&) const;

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&);

  Succeeded init_from_data_ptr(const void * const ptr)
    {
      const char * const data_ptr = reinterpret_cast<const char * const >(ptr);
      std::copy(data_ptr, data_ptr+sizeof(this->raw), reinterpret_cast<char *>(&this->raw));
      return Succeeded::yes;
    }
  inline bool is_prompt() const { return this->data.delayed == 1; }
  inline Succeeded set_prompt(const bool prompt = true) 
  { if (prompt) this->data.delayed=1; else this->data.delayed=0; return Succeeded::yes; }

 private:
  BOOST_STATIC_ASSERT(sizeof(CListEventDataECAT8_32bit)==4); 
  union 
  {
    CListEventDataECAT8_32bit   data;
    boost::int32_t         raw;
  };
  std::vector<int> segment_sequence;
  std::vector<int> sizes;

}; 


} // namespace ecat
END_NAMESPACE_STIR

#endif
