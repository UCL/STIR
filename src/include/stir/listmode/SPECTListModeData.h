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
  \brief Declaration of class stir::GSPECTListModeData

  \author Daniel Deidda
  \author Kris Thielemans
*/

#ifndef __stir_listmode_SPECTListModeData_H__
#define __stir_listmode_SPECTListModeData_H__

#include "stir/listmode/ListModeData.h"
#include "stir/listmode/SPECTListRecord.h"

# ifdef BOOST_NO_STDC_NAMESPACE
namespace std { using ::time_t; }
#endif

START_NAMESPACE_STIR

class Succeeded;
class ExamInfo;

/*!
  \brief The base class for reading SPECT list mode data.
  \ingroup listmode

  This class (and SPECTListRecord) is specific to SPECT, i.e. to
  single gamma detection. However, the only difference with the PET part
  are the functions related to prompts and delayeds.
  SPECT is handled by calling all single photon events 'prompts'.

*/
class SPECTListModeData : virtual public ListModeData
{
public:

  //! Get a pointer to an empty record
  /*! This is mainly/only useful to get a record of the correct type, that can then be
      passed to get_next_record().
  */
  virtual
    shared_ptr <SPECTListRecord> get_empty_record_sptr() const = 0;

  //! Gets the next record in the listmode sequence
  virtual
    Succeeded get_next_record(SPECTListRecord& event) const = 0;

  //! Return if the file stores delayed events as well (as opposed to prompts)
  virtual bool has_delayeds() const {return false;}

protected:

  virtual shared_ptr<ListRecord> get_empty_record_helper_sptr() const
  {
        shared_ptr<SPECTListRecord> sptr(this->get_empty_record_sptr());
        shared_ptr<ListRecord> sptr1(static_pointer_cast<ListRecord>(sptr));
        return sptr1;}

  virtual Succeeded get_next(ListRecord& event) const
  {return this->get_next_record(reinterpret_cast<SPECTListRecord&>(event));}
};

END_NAMESPACE_STIR

#endif
