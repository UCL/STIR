//
// $Id$
//
/*!
  \file
  \ingroup listmode
  \brief Declarations of classes CListRecord, CListTime and CListEvent which
  are used for list mode data.
    
  \author Kris Thielemans
      
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
    This software is distributed under the terms of the GNU Lesser General 
    Public Licence (LGPL).
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_listmode_CListRecord_H__
#define __stir_listmode_CListRecord_H__


#include "stir/round.h"
#include "stir/Succeeded.h"

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
class CListEvent
{
public:
  virtual ~CListEvent() {}

  //! Checks if this is a prompt event or a delayed event
  /*! PET scanners generally have a facility to detect events in a 
      'delayed' coincidence window. This is used to estimate the
      number of accidental coincidences (or 'randoms').
  */
  virtual
    bool
    is_prompt() const = 0;

  //! Changes the event from prompt to delayed or vice versa
  /*! Default implementation just returns Succeeded::no. */
  virtual 
    Succeeded
    set_prompt(const bool prompt = true);

  //! Finds the coordinates where the detection took place
  /*! Obviously, these coordinates are only estimates which depend on the
      scanner hardware. For example, Depth-of-Interaction might not be
      taken into account. However, the intention is that this function returns
      'likely' positions (e.g. not the face of a crystal, but a point somewhere 
      in the middle).

      Coordinates are in mm and in the standard STIR coordinate system.
    \todo This function might need time info or so for rotating scanners.
  */
  virtual
    void
    get_detection_coordinates(CartesianCoordinate3D<float>& coord_1,
			      CartesianCoordinate3D<float>& coord_2) const=0;
	
  //! Finds the LOR between the coordinates where the detection took place
  /*! Coordinates are in mm and in the standard STIR coordinate system.
      
      Implementation is in terms of get_detection_coordinates().
    \see get_detection_coordinates()
    \todo This function might need time info or so for rotating scanners.
  */
  LORAs2Points<float>
    get_LOR() const;

  //! Finds the bin coordinates of this event for some characteristics of the projection data
  /*! bin.get_bin_value() will be <=0 when the event corresponds to
      an LOR outside the range of the projection data.

      bin.get_bin_value() will be set to a negative value if no such bin
      can be found.

      Currently, bin.get_bin_value() might indicate some weight
      which can be used for normalisation. This is unlikely
      to remain the case in future versions.

      The default implementation uses get_LOR()
      and ProjDataInfo::get_bin(). However, a derived class
      can overload this with a more efficient implementation.

    \todo get_bin() might need time info or so for rotating scanners.
  */
  virtual
    void
    get_bin(Bin& bin, const ProjDataInfo&) const;

}; /*-coincidence event*/


//! A class for storing and using a timing record from a listmode file
/*! \ingroup listmode
    CListTime is used to provide an interface to the 'timing' events 
    in the list mode stream. Usually, the timing event also contains 
    gating information. For rotating scanners, it could also contain
    angle info.

    \todo this is still under development. Things to add are angles
    or so for rotating scanners. Also, some info on the maximum
    (and actual?) number of gates would be useful.
    \see CListModeData for more info on list mode data. 
*/
class CListTime
{
public:
  virtual ~CListTime() {}

  virtual unsigned long get_time_in_millisecs() const = 0;
  inline double get_time_in_secs() const
    { return get_time_in_millisecs()/1000.; }

  virtual Succeeded set_time_in_millisecs(const unsigned long time_in_millisecs) = 0;
  inline Succeeded set_time_in_secs(const double time_in_secs)
    { 
      unsigned long time_in_millisecs;
      round_to(time_in_millisecs, time_in_secs/1000.);
      return set_time_in_millisecs(time_in_millisecs); 
    }

  //! get gating info
  /*! Generally, gates are numbered from 0 to some maximum value.
   */
  virtual unsigned int get_gating() const = 0;

  virtual Succeeded set_gating(unsigned int) = 0;
};

//! A class for a general element of a list mode file
/*! \ingroup listmode
    This represents either a timing or coincidence event in a list mode
    data stream.

    Some scanners can have more types of records. For example,
    the Quad-HiDAC puts singles information in the
    list mode file. If you need that information,
    you will have to do casting to e.g. CListRecordQHiDAC.
    
    \see CListModeData for more info on list mode data. 
*/
class CListRecord
{
public:
  virtual ~CListRecord() {}

  virtual bool is_time() const = 0;

  virtual bool is_event() const = 0;

  virtual CListEvent&  event() = 0;
  virtual const CListEvent&  event() const = 0;
  virtual CListTime&   time() = 0; 
  virtual const CListTime&   time() const = 0; 

  virtual bool operator==(const CListRecord& e2) const = 0;
  bool operator!=(const CListRecord& e2) const { return !(*this == e2); }

};


END_NAMESPACE_STIR

#endif
