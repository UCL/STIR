
//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup singles_buildblock

  \brief Declaration of class stir::SinglesRates

  \author Kris Thielemans and Sanida Mustafovic
*/

#ifndef __stir_data_SinglesRates_H__
#define __stir_data_SinglesRates_H__

//#include "stir/Array.h"
#include "stir/DetectionPosition.h"
#include "stir/RegisteredObject.h"
#include "stir/Scanner.h"
#include "stir/shared_ptr.h"
#include "stir/deprecated.h"
#include <vector>

START_NAMESPACE_STIR

class TimeFrameDefinitions;


/*!
  \ingroup singles_buildblock

 \brief A single frame of singles information.

 \todo This class does not store rates, but totals! Naming is all wrong.
 \deprecated
 */
class FrameSinglesRates
{

 public:
  typedef std::vector<float>::iterator iterator;
  typedef std::vector<float>::const_iterator const_iterator;

    //! Constructor taking all arguments
    /*! \warning only checks sizes with an \c assert.
    */
    FrameSinglesRates(std::vector<float>& avg_singles_rates,
                      double start_time,
                      double end_time,
                      shared_ptr<Scanner> scanner);
    //! Constructor without singles rates
    /*! Initialises the size of the internal object that stores the singles rates
        but does not initialise its values.
    */
    FrameSinglesRates(double start_time,
                      double end_time,
                      shared_ptr<Scanner> scanner);


    //! Get singles rate for a particular singles bin index.
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    float get_singles_rate(int singles_bin_index) const;
    
    //! Get singles rate for a detection position.
    //
    // The singles rate returned is the rate for a whole singles unit.
    //
    float get_singles_rate(const DetectionPosition<>& det_pos) const;

    const_iterator begin() const
    { return this->_singles.begin(); }

    iterator begin()
    { return this->_singles.begin(); }

    const_iterator end() const
    { return this->_singles.end(); }

    iterator end()
    { return this->_singles.end(); }

    //! Get the start time of the frame whose rates are recorded.
    double get_start_time() const;
    
    //! Get the end time of the frame whose rates are recorded.
    double get_end_time() const;

    //! Get the scanner information.
    inline const Scanner * get_scanner_ptr() const;

 private:
    
    double _start_time;
    double _end_time;
    std::vector<float> _singles;
    
    // Scanner specifics
    shared_ptr<Scanner> _scanner_sptr;
    
};





/*!
  \ingroup singles_buildblock
  \brief The base-class for using singles info

  <i>Singles</i> in PET are photons detected by a single detector. In PET they
  are useful to estimate  dead-time or randoms.

  This class allows to get the singles-counts during an acquisition.
  There will be 1 per <i>singles unit</i>. See Scanner for
  some more info.

*/
class SinglesRates : public RegisteredObject<SinglesRates>
{
public: 

  virtual ~SinglesRates () {}
  //! Get the (average) singles rate for a particular singles unit and a frame with the specified start and end times.
  /*! The behaviour of this function is specified by the derived classes.
    \warning Currently might return -1 if the \a start_time, \a end_time
    are invalid (e.g. out of the measured range).

    Default implementation uses `get_singles(...)/(end_time-start_time)`.
  */
  virtual float
    get_singles_rate(const int singles_bin_index, 
		     const double start_time, 
		     const double end_time) const;

  //! Get the number of singles for a particular singles unit and a frame with the specified start and end times.
  /*! The behaviour of this function is specified by the derived classes.
    \warning Currently might return -1 if the \a start_time, \a end_time
    are invalid (e.g. out of the measured range).
  */
  virtual float
    get_singles(const int singles_bin_index,
                 const double start_time,
                 const double end_time) const = 0;
  
  //! Virtual function that returns the average singles rate given the detection positions and time-interval of detection 
  /*! The behaviour of this function is specified by the derived classes.
    \warning Currently might return -1 if the \a start_time, \a end_time
    are invalid (e.g. out of the measured range).

    Default implementation uses Scanner::get_singles_bin_index() and get_singles_rate(int,double,double).
  */
  virtual float get_singles_rate(const DetectionPosition<>& det_pos, 
				 const double start_time,
				 const double end_time) const;

  //! Virtual function that returns the number of singles given the detection positions and time-interval of detection
  /*! The behaviour of this function is specified by the derived classes.
    \warning Currently might return -1 if the \a start_time, \a end_time
    are invalid (e.g. out of the measured range).

    Default implementation uses Scanner::get_singles_bin_index() and get_singles(int,double,double).
  */

  virtual float get_singles(const DetectionPosition<>& det_pos, 
                            const double start_time,
                            const double end_time) const;

  //! Get the scanner pointer
  inline const Scanner * get_scanner_ptr() const;
  

  //! Generate a FramesSinglesRate - containing the average rates
  //  for a frame begining at start_time and ending at end_time.
  //virtual FrameSinglesRates get_rates_for_frame(double start_time,
  //                                              double end_time) const = 0;
  
#if 0
  //! return time-intervals for singles that are recorded
 virtual TimeFrameDefinitions
   get_time_frame_definitions() const = 0;
#endif

protected:
  shared_ptr<Scanner> scanner_sptr;

};








END_NAMESPACE_STIR

#include "stir/data/SinglesRates.inl"
#endif

