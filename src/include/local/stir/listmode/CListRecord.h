//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Preliminary code to handle listmode events 
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListRecord_H__
#define __stir_listmode_CListRecord_H__


#include "stir/common.h"
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::istream;
#endif

START_NAMESPACE_STIR
class Bin;
class ProjDataInfo;


//! Class for storing and using a coincidence event from a listmode file
class CListEvent
{
public:  
  virtual bool is_prompt() const = 0;
  virtual void set_prompt(const bool prompt = true) =0;
	
  virtual void get_bin(Bin&, const ProjDataInfo&) const = 0;

}; /*-coincidence event*/


//! A class for storing and using a timing 'event' from a listmode file
class CListTime
{
public:
  virtual double get_time_in_secs() const = 0;

  virtual void set_time_in_secs(const double time_in_secs) = 0;

  virtual unsigned int get_gating() const = 0;

  virtual void set_gating(unsigned int) = 0;
};

//! A class for a general element of a listmode file
class CListRecord
{
public:
  virtual bool is_time() const = 0;

  virtual bool is_event() const = 0;

  virtual CListEvent&  event() = 0;
  virtual const CListEvent&  event() const = 0;
  virtual CListTime&   time() = 0; 
  virtual const CListTime&   time() const = 0; 

  virtual bool operator==(const CListRecord& e2) const = 0;
  bool operator!=(const CListRecord& e2) const { return !(*this == e2); }
  virtual char const * get_const_data_ptr() const = 0;
  virtual char * get_data_ptr() = 0;
};


END_NAMESPACE_STIR

#endif
