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
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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

/*!
  \brief The base class for reading list mode data.
  \ingroup listmode

  \par What is list mode data?

  List mode data is a format for storing detected counts as a list, as
  opposed to a histogram. For each count, a list mode event contains
  all the properties that the scanner can give you. Hence, the
  list mode data gives you all the information about your acquisition
  that you can possibly get. Which information this is obviously depends
  on the scanner.

  For most (all?) scanners, events are stored in chronological order. 
  In addition to events, time flags are inserted into the list mode data.
  So, generally list mode data is a list of 'records', which can be
  of different types. In STIR, this concept of a 'record' corresponds to 
  the CListRecord class and its relatives (see the documentation for
  CListRecord).

  \par Usage
  
  For most applications, i.e. when one just wants to go through the list of
  all events, the code would be as follows:

  \code
  shared_ptr<CListModeData> lm_data_sptr =
    CListModeData::read_from_file(filename);

  // get a pointer to a 'template' record that will work for the scanner 
  // from which we're reading data
  shared_ptr <CListRecord> record_sptr = 
    lm_data_ptr->get_empty_record_sptr();
  // give the record a simple name to avoid cluttering the code below
  CListRecord& record = *record_sptr;

  double current_time = 0;
  while (lm_data_sptr->get_next_record(record) == Succeeded::yes) 
    {
      if (record.is_time())
      {
	current_time= record.time().get_time_in_secs();
      }
      if (record.is_event())
      {
        if (record.event().is_prompt())
	{ // do something 
        }
        // ...
      } 
    }
  \endcode

  In addition, there is a facility to 'remember' positions in the list,
  and go back to one of these positions. This could be useful to
  mark time frames. This goes as follows.

  \code
  // somehow I found out this is the start of the frame, so save it
  CListModeData::SavedPosition start_of_this_frame =
     lm_data_sptr->save_get_position();

  // now do something with this frame

  // now get back to the start of the frame
  if ( lm_data_sptr->set_get_position(start_of_this_frame)
     != Succeeded::yes)
    error("Help!");
  \endcode

  \todo Currently, this class (and CListRecord) is specific to PET, i.e. to
  coincidence detection (hence the 'C'). However, the only part that
  is not general are the functions related to prompts and delayeds.
  Potentially, we make classes ListModeData etc which would work for SPECT
  (and other count-based modalities?). Alternatively, SPECT can be
  handled by calling all single photon events 'prompts'.
  
  \par Notes for developers

  If you want to add a new type of list mode data, you have to make corresponding
  derived classes of CListModeData, CListRecord etc. You also have to modify
  CListModeData::read_from_file to recognise your data.

  \todo Use registry-based implementation of read_from_file.
*/
class CListModeData
{
public:
  //! Use this typedef for save/set_get_position
  typedef unsigned int SavedPosition;

  //! Attempts to get a CListModeData object from a file
  /*! Note that the exact format of the \a filename is determined by
      the derived class.
  */
  static CListModeData* read_from_file(const string& filename);

  //! Default constructor
  CListModeData();

  virtual
    ~CListModeData();

  //! Get a pointer to an empty record
  /*! This is mainly/only useful to get a record of the correct type, that can then be
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
  /*!
      Note that the return value is not related to the number of events
      already read. In particular, you cannot do any arithmetic on it to 
      skip a few events. This is different from e.g. std::streampos.

      \warning There is a maximum number of times this function can be called.
      This is determined by the SavedPosition type. Once you save more
      positions, the first positions will be overwritten. There is currently 
      no way of finding out after how many times this will happen (but it's
      a large number...).

      \warning These saved positions are only valid for the lifetime of the 
      CListModeData object.

      \warning A derived class might disable this facility. It will/should
      then always return Succeeded::no when calling set_get_position().
  */
  virtual
    SavedPosition save_get_position() = 0;

  //! Set the position for reading to a previously saved point
  
  virtual
    Succeeded set_get_position(const SavedPosition&) = 0;

  //! Get scanner pointer  
  /*! Returns a pointer to a scanner object that is appropriate for the 
      list mode data that is being read.
  */
  const Scanner* get_scanner_ptr() const;

  //! Return if the file stores delayed events as well (as opposed to prompts)
  virtual bool has_delayeds() const = 0;

protected:
  //! Has to be set by the derived class
  shared_ptr<Scanner> scanner_ptr;
};

END_NAMESPACE_STIR

#endif
