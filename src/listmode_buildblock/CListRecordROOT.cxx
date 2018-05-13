/*
    Copyright (C) 2015-2016 University of Leeds
    Copyright (C) 2016 UCL
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
  \brief Implementation of classes stir::ecat::CListEventROOT and stir::ecat::CListRecordROOT
  for listmode events for the ROOT as listmode file format.

  \author Nikos Efthimiou
  \author Harry Tsoumpas
*/

#include "stir/listmode/CListRecordROOT.h"

START_NAMESPACE_STIR

CListRecordROOT::
CListRecordROOT(const shared_ptr<Scanner> &scanner_sptr) :
    CListRecord()
{
    event_data = new CListEventROOT(scanner_sptr);
    time_data = new CListTimeROOT();
}

Succeeded
CListRecordROOT::init_from_data( const int& ring1,
                                  const int& ring2,
                                  const int& crystal1,
                                  const int& crystal2,
                                  double time1, double time2,
                                  const int& event1, const int& event2)
{
    /// \warning ROOT data are time and event at the same time.

    event_data->init_from_data(ring1, ring2,
                                    crystal1, crystal2);

     time_data->init_from_data(
                time1,time2);

    // We can make a singature raw based on the two events IDs.
    // It is pretty unique.
    raw[0] = event1;
    raw[1] = event2;

    return Succeeded::yes;
}

bool CListRecordROOT::is_time() const
{ return true; }

bool CListRecordROOT::is_event() const
{ return true; }


END_NAMESPACE_STIR
