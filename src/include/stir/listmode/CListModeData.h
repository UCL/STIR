
/*
  Copyright (C) 2019, National Physical Laboratory
  Copyright (C) 2019, University College of London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup listmode
  \brief Declaration of class stir::CListModeData
    
\author Daniel Deidda
\author Kris Thielemans
*/

#ifndef __stir_listmode_CListModeData_H__
#define __stir_listmode_CListModeData_H__

#include <string>
#include <ctime>
#include "stir/ProjDataInfo.h"
#include "stir/ExamData.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/listmode/ListModeData.h"
#include "stir/listmode/CListRecord.h"

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; }
#endif

START_NAMESPACE_STIR
class CListRecord;
class Succeeded;
class ExamInfo;

/*!
  \brief The base class for reading PET (i.e. coincidence) list mode data.
  \ingroup listmode

  The only difference w.r.t. ListModeData is the used of CListRecord and
  a virtual function to check if delayeds are present.
*/
class CListModeData : public ListModeData
{
public:
  //! Get a pointer to an empty record
  /*! This is mainly/only useful to get a record of the correct type, that can then be
      passed to get_next_record().
  */
  virtual
    shared_ptr <CListRecord> get_empty_record_sptr() const = 0;

  //! Gets the next record in the listmode sequence

    virtual
    Succeeded get_next_record(CListRecord& event) const = 0;

  //! Return if the file stores delayed events as well (as opposed to prompts)
  bool has_delayeds() const override = 0;

protected:
  shared_ptr<ListRecord> get_empty_record_helper_sptr() const override
  {
        shared_ptr<CListRecord> sptr(this->get_empty_record_sptr());
        shared_ptr<ListRecord> sptr1(static_pointer_cast<ListRecord>(sptr));
        return sptr1;}

  Succeeded get_next(ListRecord& event) const override
  {return this->get_next_record((CListRecord&)(event));}
};

END_NAMESPACE_STIR

#endif
