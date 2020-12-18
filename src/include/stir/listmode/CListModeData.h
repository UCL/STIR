
/*
  Copyright (C) 2019, National Physical Laboratory
  Copyright (C) 2019, University College of London
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
  \brief The base class for reading list mode data.
  \ingroup listmode

  \todo Currently, this class (and CListRecord) is specific to PET, i.e. to
  coincidence detection (hence the 'C'). However, the only part that
  is not general are the functions related to prompts and delayeds.
  Potentially, we make classes ListModeData etc which would work for SPECT
  (and other count-based modalities?). Alternatively, SPECT can be
  handled by calling all single photon events 'prompts'.
  
  \par Notes for developers
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
  virtual bool has_delayeds() const = 0;

protected:
  virtual shared_ptr<ListRecord> get_empty_record_helper_sptr() const
  {
        shared_ptr<CListRecord> sptr(this->get_empty_record_sptr());
        shared_ptr<ListRecord> sptr1(static_pointer_cast<ListRecord>(sptr));
        return sptr1;}

  virtual Succeeded get_next(ListRecord& event) const
  {return this->get_next_record((CListRecord&)(event));}
};

END_NAMESPACE_STIR

#endif
