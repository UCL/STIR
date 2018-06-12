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
  \brief Classes for listmode events for GATE simulated ROOT data

  \author Efthimiou Nikos
  \author Harry Tsoumpas
*/

#ifndef __stir_listmode_CListRecordROOT_H__
#define __stir_listmode_CListRecordROOT_H__

#include "stir/listmode/CListRecord.h"
#include "stir/listmode/CListEventROOT.h"
#include "stir/listmode/CListTimeROOT.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

using namespace std;

//! A class for a general element of a listmode file for a Siemens scanner using the ROOT files
class CListRecordROOT : public CListRecord // currently no gating yet
{
public:
    //! Returns always true
    bool is_time() const;
    //! Returns always true
    bool is_event() const;
    //! Returns always true
    inline bool is_full_event() const;

    virtual CListEventROOT&  event();

    virtual const CListEventROOT& event() const;

    virtual CListTimeROOT& time();

    virtual const CListTimeROOT& time() const;

    bool operator==(const CListRecord& e2) const
    {
        return dynamic_cast<CListRecordROOT const *>(&e2) != 0 &&
                raw[0] == dynamic_cast<CListRecordROOT const &>(e2).raw[0] &&
                raw[1] == dynamic_cast<CListRecordROOT const &>(e2).raw[1];
    }

    CListRecordROOT(const shared_ptr<Scanner>& scanner_sptr);

    virtual Succeeded init_from_data( const int& ring1,
                                      const int& ring2,
                                      const int& crystal1,
                                      const int& crystal2,
                                      double time1, double time2,
                                      const int& event1, const int& event2);
private:
    CListEventROOT  event_data;
    CListTimeROOT  time_data;
    boost::int32_t raw[2]; // this raw field isn't strictly necessary, get rid of it?

};

END_NAMESPACE_STIR
#include "stir/listmode/CListRecordROOT.inl"
#endif

