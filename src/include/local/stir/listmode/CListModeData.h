//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Declaration of class CListModeData
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListModeData_H__
#define __stir_listmode_CListModeData_H__

#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::size_t;
#endif

START_NAMESPACE_STIR
class CListRecord;
class Succeeded;

class CListModeData
{
public:
  //! Use this type for save/set_get_position
  typedef unsigned int SavedPosition;

  //! Attempts to get a CListModeData object from a file
  static CListModeData* read_from_file(const string& filename);

  //! Default constructor
  CListModeData();

  virtual
    ~CListModeData();

  //! Get a pointer to an empty record
  /*! This is mainly useful to get a record of the correct derived type, that can then be
      passed to get_next_record().
  */
  virtual
    shared_ptr <CListRecord> get_empty_record_sptr() const = 0;

  //! Gets the next record in the listmode sequence
  virtual 
    Succeeded get_next_record(CListRecord& event) const = 0;

  //! Call this function if you want to re-start reading at the beginning.
  virtual 
    Succeeded reset() = 0;

  //! Save the current reading position
  /*! \warning There is a maximum number of times this function will be called.
      This is determined by the SavedPosition type. Once you save more
      positions, the first positions will be overwritten.
      \warning These saved positions are only valid for the lifetime of the 
      CListModeData object.
  */
  virtual
    SavedPosition save_get_position() = 0;

  //! Set the position for reading to a previously saved point
  virtual
    Succeeded set_get_position(const SavedPosition&) = 0;

  //! Get scanner pointer  
  const Scanner* get_scanner_ptr() const;

  //! Return if the file stores delayed events as well (as opposed to prompts)
  virtual bool has_delayeds() const = 0;

protected:
  //! Has to be set by the derived class
  shared_ptr<Scanner> scanner_ptr;
};

END_NAMESPACE_STIR

#endif
