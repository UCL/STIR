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
    Copyright (C) University College London, 2017, 2023, 2026
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

//! Base-class for storing and using a coincidence event for a list-mode file that uses detector indices
/*! \ingroup listmode
    For scanners with discrete detectors, the list mode events usually store detector indices
    in some way. This class provides virtual members to to set/get those detection indices
    via DetectionPositionPair.

    \see CListEventScannerWithDiscreteDetectors. This base-class exists for cases where we don't
    want/need to know the ProjDataInfo type.
*/
class CListEventScannerWithDiscreteDetectorsBase : public CListEvent
{
public:
  //! This routine returns the corresponding detector pair
  virtual void get_detection_position(DetectionPositionPair<>&) const = 0;

  //! This routine sets in a coincidence event from detector "indices"
  virtual void set_detection_position(const DetectionPositionPair<>&) = 0;
};

//! Class for coincidence events from a list-mode file that uses detector indices and a specific type of ProjDataInfo
/*! \ingroup listmode
    This class provides more efficient implementations of some virtual members of CListEvent.
*/
template <class ProjDataInfoT>
class CListEventScannerWithDiscreteDetectors : public CListEventScannerWithDiscreteDetectorsBase
{
public:
  explicit CListEventScannerWithDiscreteDetectors(const shared_ptr<const ProjDataInfo>& proj_data_info);

  const Scanner* get_scanner_ptr() const { return this->uncompressed_proj_data_info_sptr->get_scanner_ptr(); }

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

private:
  shared_ptr<const ProjDataInfoT> uncompressed_proj_data_info_sptr;
};

END_NAMESPACE_STIR

#include "stir/listmode/CListEventScannerWithDiscreteDetectors.inl"

#endif
