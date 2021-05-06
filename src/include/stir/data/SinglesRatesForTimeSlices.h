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

  \brief Declaration of class stir::SinglesRatesForTimeSlices

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Tim Borgeaud

*/

#ifndef __stir_data_SinglesRatesForTimeSlices_H__
#define __stir_data_SinglesRatesForTimeSlices_H__

#include "stir/data/SinglesRates.h"
#include "stir/Array.h"
#include "stir/TimeFrameDefinitions.h"


START_NAMESPACE_STIR

/*!
  \ingroup singles_buildblock
  \brief A class for singles that are recorded at equal time intervals

*/
class SinglesRatesForTimeSlices : 
public SinglesRates

{ 
public:

 //! Default constructor 
 SinglesRatesForTimeSlices();

 // implementation of pure virtual in SinglesRates
 virtual float
   get_singles(const int singles_bin_index,
               const double start_time, const double end_time) const;


 //! Generate a FramesSinglesRate - containing the average rates
 //  for a frame begining at start_time and ending at end_time.
 FrameSinglesRates STIR_DEPRECATED get_rates_for_frame(double start_time,
                                       double end_time) const;


 /*
  *! Get time slice index for a time slice ending at or after t.
  *  
  * Each slice of singles data has a corresponing time recorded with
  * the singles counts. This time is considered to represent the time
  * at the end of the slice.
  * 
  * For a given double precision number of seconds, t, this function
  * will return the slice index for the first time slice that has a 
  * corresponding time greater than or equal to t.
  *
  * Assuming contiguous slices that end at the time recorded for the slice,
  * this function returns the slice in which t is contained.
  *
  * This function assumes that all slices are continguous.
  * If the supplied t does not actually fall within a frame, the closest
  * frame (ending after t) is returned. Values of t before the first time 
  * slice will result in the index to the first slice being returned.
  */
 virtual int get_end_time_slice_index(double t) const;
 
 
 /*
  *! Get time slice index for a time slice ending after t.
  *  
  * Each slice of singles data has a corresponing time recorded with
  * the singles counts. This time is considered to represent the time
  * at the end of the slice.
  * 
  * For a given double precision number of seconds, t, this function
  * will return the slice index for the first time slice that has a  
  * correspdoning time greater than t.
  *
  * Assuming contiguous slices that end at the time recorded for the slice,
  * this function returns the slice which starts before t. A time interval
  * that begins at t should contain only time slices that end _after_ t
  * not at t.
  *
  * This function assumes that all slices are continguous.
  * If the supplied t does not actually fall within a frame, the closest
  * frame (ending after t) is returned. Values of t before the first time 
  * slice will result in the index to the first slice being returned.
  */
 virtual int get_start_time_slice_index(double t) const;


 #if 0
 //! Get rates using time slice and singles bin indices.
 //
 // The singles rate returned is the rate for a whole singles unit.
 //
 int get_singles_rate(int singles_bin_index, int time_slice) const;
#endif
 //! Set a singles by singles bin index and time slice.
 void set_singles(int singles_bin_index, int time_slice, int new_singles);


 //! Rebin the sgl slices into a different set of consecutive slices.
 //
 // Returns the number of new bins.
 int rebin(std::vector<double>& new_end_times);
 
  
 //! Get the vector of time values for each time slice index.
 std::vector<double> get_times() const;


 // Some inspectors

 //! Return the number of time slices.
 int get_num_time_slices() const;

 
 //! Return the time interval per slice of singles data.
 double get_singles_time_interval() const;

 //! return time-intervals for every slice
 TimeFrameDefinitions
   get_time_frame_definitions() const;


protected:
 
 //! total singles per time slice and singles-bin
 /*!Indexed by time slice and singles bin index.*/
 Array<2, int> _singles;

 //! end times of each time slice (in secs)
 /*! expected to use equidistant sampling */
 std::vector<double> _times;
 
 int _num_time_slices;

 //! time interval in secs
 /*! \warning A value of zero for _singles_time_interval indicates that the time slices
   are of different lengths.
   However, some of the code probably doesn't check for this.
 */
 double _singles_time_interval;

 //! Calculate and set _singles_time_interval from _times
 void set_time_interval();

 //! get slice start time.
 double get_slice_start_time(int slice_index) const;
 
};

END_NAMESPACE_STIR


#endif
