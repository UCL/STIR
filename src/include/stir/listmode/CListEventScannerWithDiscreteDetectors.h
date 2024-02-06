//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of class stir::CListEventScannerWithDiscreteDetectors

  \author Kris Thielemans

*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_listmode_CListEventScannerWithDiscreteDetectors_H__
#define __stir_listmode_CListEventScannerWithDiscreteDetectors_H__

#include "stir/Succeeded.h"
#include "stir/DetectionPositionPair.h"
#include "stir/ProjDataInfo.h"
#include "stir/listmode/CListRecord.h"

START_NAMESPACE_STIR

//! Class for storing and using a coincidence event from a list mode file for a cylindrical single layer scanner
/*! \ingroup listmode
    For scanners with discrete detectors, the list mode events usually store detector indices
    in some way. This class provides access mechanisms to those detection positions, and
    also provides more efficient implementations of some virtual members of CListEvent.
*/
template <class ProjDataInfoT>
class CListEventScannerWithDiscreteDetectors : public CListEvent
{
public:
  explicit CListEventScannerWithDiscreteDetectors(const shared_ptr<const ProjDataInfo>& proj_data_info);

  const Scanner* get_scanner_ptr() const { return this->uncompressed_proj_data_info_sptr->get_scanner_ptr(); }

  //! This routine returns the corresponding detector pair
  virtual void get_detection_position(DetectionPositionPair<>&) const = 0;

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&) = 0;

  //! find LOR between detector pairs
  /*! Overrides the default implementation to use get_detection_position()
    which should be faster.
  */
  inline LORAs2Points<float> get_LOR() const override;

  //! find bin for this event
  /*! Overrides the default implementation to use get_detection_position()
    which should be faster.

    \warning This implementation is only valid for \c proj_data_info of
    type ProjDataInfoT. However, because of efficiency reasons
    this is only checked in debug mode (NDEBUG not defined).
  */
  inline void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const override;

  //! This method checks if the template is valid for LmToProjData
  /*! Used before the actual processing of the data (see issue #61), before calling get_bin()
   *  Most scanners have listmode data that correspond to non arc-corrected data and
   *  this check avoids a crash when an unsupported template is used as input.
   */
  inline bool is_valid_template(const ProjDataInfo&) const override;

protected:
  shared_ptr<const ProjDataInfoT> get_uncompressed_proj_data_info_sptr() const { return uncompressed_proj_data_info_sptr; }

  // shared_ptr<Scanner> scanner_sptr;

private:
  shared_ptr<const ProjDataInfoT> uncompressed_proj_data_info_sptr;
};

END_NAMESPACE_STIR

#include "stir/listmode/CListEventScannerWithDiscreteDetectors.inl"

#endif
