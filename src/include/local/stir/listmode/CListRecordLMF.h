//
// $Id$
//
/*!
  \file
  \ingroup ClearPET_utilities
  \brief Preliminary code to handle listmode events 
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd

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
  { return Succeeded::yes; }

  void
    get_detection_coordinates(CartesianCoordinate3D<float>& coord_1,
			      CartesianCoordinate3D<float>& coord_2) const;

  CartesianCoordinate3D<float> pos1() const
    { return detection_pos1; }
  CartesianCoordinate3D<float>& pos1()
    { return detection_pos1; }
  CartesianCoordinate3D<float> pos2() const
    { return detection_pos2; }
  CartesianCoordinate3D<float>& pos2()
    { return detection_pos2; }

 private:
  CartesianCoordinate3D<float> detection_pos1;
  CartesianCoordinate3D<float> detection_pos2;
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
  inline unsigned int get_gating() const //TODO
  { return gating; }
  inline Succeeded set_gating(unsigned int g)//TODO
  { return Succeeded::no;}
private:
  unsigned long time; // in millisecs TODO
  unsigned    gating;
};


//! A class for a general element of a listmode file
/*! \ingroup ClearPET_utilities

I really have no clue how time info is handled in LMF. At the moment,
I store both time and CListEvent info in one CListRecordLMF.
That's obviously not necessary nor desirable.

Better would be to have some kind of union, after reading check what
event type we have, and then fill in the appropriate members (in a union?).
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


  virtual
    void
    get_detection_coordinates(CartesianCoordinate3D<float>& coord_1,
			      CartesianCoordinate3D<float>& coord_2) const
  { 
    event_data.get_detection_coordinates(coord_1, coord_2);
  }
 private:
  friend class CListModeDataLMF;

  CListEventDataLMF  event_data;
  CListTimeDataLMF   time_data; 

  bool is_time_flag;
};



END_NAMESPACE_STIR

#endif
