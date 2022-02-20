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
    Copyright (C) 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_listmode_CListEventCylindricalScannerWithDiscreteDetectors_H__
#define __stir_listmode_CListEventCylindricalScannerWithDiscreteDetectors_H__

#include "stir/Succeeded.h"
#include "stir/DetectionPositionPair.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoBlocksOnCylindricalNoArcCorr.h"
#include "stir/ProjDataInfoGenericNoArcCorr.h"
#include "stir/listmode/CListRecord.h"
#include <memory>

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

  //! This method checks if the template is valid for LmToProjData
  /*! Used before the actual processing of the data (see issue #61), before calling get_bin()
   *  For instance, most scanners have listmode data that correspond to non arc-corrected data and
   *  this check avoids a crash when an unsupported template is used as input.
   */
  inline virtual bool is_valid_template(const ProjDataInfo&) const;

 protected:
   template <typename T>
   shared_ptr<const T>
    get_proj_data_info_sptr_cast() const
     {
       assert(dynamic_cast<const T*>(this->proj_data_info_sptr.get()) != 0);
       return std::static_pointer_cast<const T>(this->proj_data_info_sptr);
     }
   //! legacy version for the cylindrical case
   shared_ptr<const ProjDataInfoCylindricalNoArcCorr>
    get_uncompressed_proj_data_info_sptr() const
     {
       return get_proj_data_info_sptr_cast<ProjDataInfoCylindricalNoArcCorr>();
     }
   shared_ptr<const ProjDataInfo>
    get_proj_data_info_sptr() const
     {
       return this->proj_data_info_sptr;
     }

   shared_ptr<Scanner> scanner_sptr;
 private:
   shared_ptr<const ProjDataInfo> proj_data_info_sptr;
   enum scanner_type {CYL, BLOCK, GEN};
   scanner_type scanner_type;
};

END_NAMESPACE_STIR

#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.inl"

#endif
