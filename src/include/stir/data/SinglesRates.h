
//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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

#include <vector>

START_NAMESPACE_STIR






/*!
  \ingroup singles_buildblock

 \brief A single frame of singles information.
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
  are useful to estimate  dead-time.

  This class allows to get the rate of singles during an acquisition.
  There will be 1 rate per <i>singles unit</i>. See Scanner for
  some more info.
*/
class SinglesRates : public RegisteredObject<SinglesRates>
{
public: 

  virtual ~SinglesRates () {}
  //! Get the singles rate for a particular singles unit and a frame with the specified start and end times.   
  /*! The behaviour of this function is specified by the derived classes.
    \warning Currently might return -1 if the \a start_time, \a end_time
    are invalid (e.g. out of the measured range).
  */
  virtual float
    get_singles_rate(const int singles_bin_index, 
		     const double start_time, 
		     const double end_time) const = 0;
  
  //! Virtual function that returns the average singles rate given the detection positions and time-interval of detection 
  /*! The behaviour of this function is specified by the derived classes.
    \warning Currently might return -1 if the \a start_time, \a end_time
    are invalid (e.g. out of the measured range).

    Default implementation uses Scanner::get_singles_bin_index().
  */

  virtual float get_singles_rate(const DetectionPosition<>& det_pos, 
				 const double start_time,
				 const double end_time) const;
  
  //! Get the scanner pointer
  inline const Scanner * get_scanner_ptr() const;
  

  //! Generate a FramesSinglesRate - containing the average rates
  //  for a frame begining at start_time and ending at end_time.
  //virtual FrameSinglesRates get_rates_for_frame(double start_time,
  //                                              double end_time) const = 0;
  
  
protected:
  shared_ptr<Scanner> scanner_sptr;

};








END_NAMESPACE_STIR

#include "stir/data/SinglesRates.inl"
#endif

