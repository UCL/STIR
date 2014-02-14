//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of class stir::CListEventCylindricalScannerWithDiscreteDetectors
    
  \author Kris Thielemans
      
*/
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
#ifndef __stir_listmode_CListEventCylindricalScannerWithDiscreteDetectors_H__
#define __stir_listmode_CListEventCylindricalScannerWithDiscreteDetectors_H__

#include "stir/Succeeded.h"
#include "stir/DetectionPositionPair.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"

START_NAMESPACE_STIR

//! Class for storing and using a coincidence event from a list mode file for a cylindrical single layer scanner
/*! \ingroup listmode
    For scanners with discrete detectors, the list mode events usually store detector indices
    in some way. This class provides access mechanisms to those detection positions, and
    also provides more efficient implementations of some virtual members of CListEvent.
*/
class CListEventCylindricalScannerWithDiscreteDetectors : public CListEvent
{
public:
  inline explicit 
    CListEventCylindricalScannerWithDiscreteDetectors(const shared_ptr<Scanner>& scanner_sptr);

  const Scanner * get_scanner_ptr() const
    { return this->scanner_sptr.get(); }

  //! This routine returns the corresponding detector pair   
  virtual void get_detection_position(DetectionPositionPair<>&) const = 0;

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&) = 0;

  //! find LOR between detector pairs
  /*! Overrides the default implementation to use get_detection_position()
    which should be faster.
  */
  inline virtual LORAs2Points<float> get_LOR() const;

  //! find bin for this event
  /*! Overrides the default implementation to use get_detection_position()
    which should be faster.

    \warning This implementation is only valid for \c proj_data_info of 
    type ProjDataInfoCylindricalNoArcCorr. However, because of efficiency reasons
    this is only checked in debug mode (NDEBUG not defined).
  */
  inline virtual void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const;

 protected:
   shared_ptr<ProjDataInfoCylindricalNoArcCorr>
    get_uncompressed_proj_data_info_sptr() const
     {
       return uncompressed_proj_data_info_sptr;
     }

   shared_ptr<Scanner> scanner_sptr;
 private:
   shared_ptr<ProjDataInfoCylindricalNoArcCorr>
     uncompressed_proj_data_info_sptr;

};

END_NAMESPACE_STIR

#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.inl"

#endif
