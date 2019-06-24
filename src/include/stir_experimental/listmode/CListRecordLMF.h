//
//
/*!
  \file
  \ingroup ClearPET_utilities
  \brief Preliminary code to handle listmode events 
    
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2003- 2011, Hammersmith Imanet Ltd

    This software is distributed under the terms 
    of the GNU Lesser General  Public Licence (LGPL)
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListRecordLMF_H__
#define __stir_listmode_CListRecordLMF_H__

#include "stir/listmode/CListRecord.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

class CListModeDataLMF;

//! Class for storing and using a coincidence event from a listmode file
/*! \ingroup ClearPET_utilities
 */
class CListEventDataLMF 
{
 public:  
  inline bool is_prompt() const { return true; } // TODO
  inline Succeeded set_prompt(const bool prompt = true) // TODO
  { return Succeeded::no; }
  
  inline LORAs2Points<float> get_LOR() const
    { return this->lor; }


  CartesianCoordinate3D<float> pos1() const
    { return lor.p1(); }
  CartesianCoordinate3D<float>& pos1()
    { return lor.p1(); }
  CartesianCoordinate3D<float> pos2() const
    { return lor.p2(); }
  CartesianCoordinate3D<float>& pos2()
    { return lor.p1(); }

 private:
  LORAs2Points<float> lor;
}; /*-coincidence event*/

class CListRecordLMF;

//! A class for storing and using a timing 'event' from a listmode file
/*! \ingroup ClearPET_utilities
 */
class CListTimeDataLMF
{
 public:
  inline unsigned long get_time_in_millisecs() const
    { return time;  }// TODO
  inline Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs)//TODO
  { return Succeeded::no; }
private:
  unsigned long time; // in millisecs TODO
};


//! A class for a general element of a listmode file
/*! \ingroup ClearPET_utilities

I really have no clue how time info is handled in LMF. At the moment,
I store both time and CListEvent info in one CListRecordLMF.
That's obviously not necessary nor desirable.

 */
class CListRecordLMF : public CListRecord, public CListTime, public CListEvent
{
public:

  CListRecordLMF& operator=(const CListEventDataLMF& event)
    {
      is_time_flag=false;
      event_data = event;
    }
  bool is_time() const
  { return is_time_flag == true; }
  bool is_event() const
  { return is_time_flag == false; }
  virtual CListEvent&  event() 
    { return *this; }
  virtual const CListEvent&  event() const
    { return *this; }
  virtual CListTime&   time()
    { return *this; }
  virtual const CListTime&   time() const
    { return *this; }

  bool operator==(const CListRecord& e2) const;

  // time 
  inline double get_time_in_secs() const 
    { return time_data.get_time_in_secs(); }
  inline Succeeded set_time_in_secs(const double time_in_secs)
    { return time_data.set_time_in_secs(time_in_secs); }
  inline unsigned int get_gating() const
    { return time_data.get_gating(); }
  inline Succeeded set_gating(unsigned int g) 
    { return time_data.set_gating(g); }

  // event
  inline bool is_prompt() const { return event_data.is_prompt(); }
  inline Succeeded set_prompt(const bool prompt = true) 
  { return event_data.set_prompt(prompt); }


 private:
  friend class CListModeDataLMF;

  CListEventDataLMF  event_data;
  CListTimeDataLMF   time_data; 

  bool is_time_flag;
};



END_NAMESPACE_STIR

#endif
