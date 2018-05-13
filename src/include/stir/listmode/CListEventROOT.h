/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
    Copyright (C) 2018 University of Hull
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
  \brief Classes for listmode events for GATE simulated ROOT data

  \author Efthimiou Nikos
  \author Harry Tsoumpas
*/

#ifndef __stir_listmode_CListEventROOT_H__
#define __stir_listmode_CListEventROOT_H__

#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"

START_NAMESPACE_STIR

class CListEventROOT : public CListEventCylindricalScannerWithDiscreteDetectors
{
public:

    CListEventROOT(const shared_ptr<Scanner>& scanner_sptr);

    //! This routine returns the corresponding detector pair
    virtual void get_detection_position(DetectionPositionPair<>&) const;

    //! This routine sets in a coincidence event from detector "indices"
    virtual void set_detection_position(const DetectionPositionPair<>&);

    //! This is the main function which transform GATE coordinates to STIR
    inline void init_from_data(const int &_ring1, const int &_ring2,
                             const int &crystal1, const int &crystal2);

    bool is_prompt() const;

    bool inline is_swapped() const
    { return swapped; }

private:
    //! First ring, in order to detector tangestial index
    int ring1;
    //! Second ring, in order to detector tangestial index
    int ring2;
    //! First detector, in order to detector tangestial index
    int det1;
    //! Second detector, in order to detector tangestial index
    int det2;
    //! Indicates if swap segments
    bool swapped;
    //! This is the number of detector we have to rotate in order to
    //! align GATE and STIR.
    int quarter_of_detectors;
};

END_NAMESPACE_STIR
#include "stir/listmode/CListEventROOT.inl"

#endif


