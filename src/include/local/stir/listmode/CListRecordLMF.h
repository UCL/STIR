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
    Copyright (C) 2003- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListRecordLMF_H__
#define __stir_listmode_CListRecordLMF_H__

#include "local/stir/listmode/CListRecord.h"
#include "stir/ProjDataInfoCylindrical.h"
#include "stir/CartesianCoordinate3D.h"


START_NAMESPACE_STIR


//! Class for storing and using a coincidence event from a listmode file
*/
class CListEventDataLMF 
{
 public:  
  inline bool is_prompt() const { return true; } // TODO
  inline void set_prompt(const bool prompt = true) // TODO
    { }

  void get_bin(Bin&, const ProjDataInfo&) const;

  CartesianCoordinate3D<double> pos1() const
    { return detection_pos1; }
  CartesianCoordinate3D<double>& pos1()
    { return detection_pos1; }
  CartesianCoordinate3D<double> pos2() const
    { return detection_pos2; }
  CartesianCoordinate3D<double>& pos2()
    { return detection_pos2; }

 private:
  CartesianCoordinate3D<double> detection_pos1;
  CartesianCoordinate3D<double> detection_pos2;
}; /*-coincidence event*/

class CListRecordLMF;

//! A class for storing and using a timing 'event' from a listmode file
class CListTimeDataLMF
{
 public:
  inline double get_time_in_secs() const
    { return time/1000.;  }// TODO
  inline void set_time_in_secs(const double time_in_secs)
    { time = time_in_secs*1000; }
  inline unsigned int get_gating() const
  { return gating; }
  inline void set_gating(unsigned int g)
    { gating = g;}
private:
  double time; // in millisecs TODO
  unsigned    gating;
};


//! A class for a general element of a listmode file
  class CListRecordLMF : public CListRecord, public CListTime, public CListEvent
{
public:

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

  virtual char const * get_const_data_ptr() const
    { return reinterpret_cast<char const *>(&raw); }// TODO
  virtual char * get_data_ptr() 
    { return reinterpret_cast<char *>(&raw); }

  // time 
  inline double get_time_in_secs() const 
    { return time_data.get_time_in_secs(); }
  inline void set_time_in_secs(const double time_in_secs)
    { time_data.set_time_in_secs(time_in_secs); }
  inline unsigned int get_gating() const
    { return time_data.get_gating(); }
  inline void set_gating(unsigned int g) 
    { time_data.set_gating(g); }

  // event
  inline bool is_prompt() const { return event_data.is_prompt(); }
  inline void set_prompt(const bool prompt = true) { event_data.set_prompt(prompt); }
  void get_bin(Bin& bin, const ProjDataInfo& proj_data_info) const 
    {
      event_data.get_bin(bin, dynamic_cast<const ProjDataInfoCylindrical&>(proj_data_info));
    }
  // private:
  union {
    CListEventDataLMF  event_data;
    CListTimeDataLMF   time_data; 
  };
  bool is_time_flag;
};



END_NAMESPACE_STIR

#endif
