//
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes stir::CListRecord, and stir::CListEvent which
  are used for list mode data.
    
  \author Daniel Deidda
  \author Kris Thielemans

*/
/*
    Copyright (C) 2019, National Physical Laboratory
    Copyright (C) 2019, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListRecord_H__
#define __stir_listmode_CListRecord_H__

#include "ListRecord.h"

START_NAMESPACE_STIR
class Bin;
class ProjDataInfo;
class Succeeded;
template <typename coordT> class CartesianCoordinate3D;
template <typename coordT> class LORAs2Points;

//! Class for storing and using a coincidence event from a list mode file
/*! \ingroup listmode
    CListEvent is used to provide an interface to the actual events (i.e.
    detected counts) in the list mode stream.

    \todo this is still under development. Things to add are for instance
    energy windows and time-of-flight info. Also, get_bin() would need
    time info or so for rotating scanners.

    \see CListModeData for more info on list mode data. 
*/
class CListEvent : public ListEvent
{
public:

  //! Changes the event from prompt to delayed or vice versa
  /*! Default implementation just returns Succeeded::no. */
  virtual 
    Succeeded
    set_prompt(const bool prompt = true);

}; /*-coincidence event*/

class CListRecord : public ListRecord
{
public:
};

class CListRecordWithGatingInput : public CListRecord
{};

END_NAMESPACE_STIR

#endif
