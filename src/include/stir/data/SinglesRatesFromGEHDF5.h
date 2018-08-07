//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    Copyright (C) 2017-2018, University of Leeds
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

  \brief Declaration of class stir::SinglesRatesFromGEHDF5

  \author Palak Wadhwa
  \author Kris Thielemans

*/

#ifndef __stir_data_SinglesRatesFromGEHDF5_H__
#define __stir_data_SinglesRatesFromGEHDF5_H__

#include "stir/data/SinglesRates.h"
#include "stir/Array.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/IO/HDF5Wrapper.h"


START_NAMESPACE_STIR



/*!
  \ingroup singles_buildblock
  \brief A class for reading singles over the number of time samples from an GE HDF5 .BLF listmode file format.

  .BLF files are generated as a result of PET scan by GE SIGNA PET/MR scanners.
*/
class SinglesRatesFromGEHDF5 : 
public RegisteredParsingObject<SinglesRatesFromGEHDF5, SinglesRates>

{ 
public:

 //! Name which will be used when parsing a SinglesRatesFromGEHDF5 object 
 static const char * const registered_name; 

//PW Would not touch this.
 //! Default constructor 
 SinglesRatesFromGEHDF5();

 // implementation of pure virtual in SinglesRates
 virtual float
   get_singles_rate(const int singles_bin_index, 
		    const double start_time, const double end_time) const;

 //! Generate a FramesSinglesRate - containing the average rates
 //  for a frame begining at start_time and ending at end_time.
 FrameSinglesRates get_rates_for_frame(double start_time,
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


 //! Get rates using time slice and singles bin indices.
 //
 // The singles rate returned is the rate for a whole singles unit.
 //
 int get_singles_rate(int singles_bin_index, int time_slice) const;

 float get_singles_rate(const DetectionPosition<>& det_pos,
                 const double start_time,
                 const double end_time) const;

 //! Set a singles rate by singles bin index and time slice.
 //
 // The singles rate returned is the rate for a whole singles unit.
 //
 void set_singles_rate(int singles_bin_index, int time_slice, int new_rate);


 //! Rebin the .BLF slices into a different set of consecutive slices.
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

 
 // IO Methods
//PW Reading singles from .sgl changed to .BLF file format. Adapt from GE HDF5 listmode file read.
 //! The function that reads singles from *.sgl file.
 int read_singles_from_listmode_file(const std::string& _listmode_filename);

 /*!
  * \brief Write the SinglesRatesFromGEHDF5 object to a stream.
  * \param[in] output The ostream to which the object will be written.
  */
 //PW Here writing of singles in output stream; unsure
 std::ostream& write(std::ostream& output);

 

private:
 
 // Indexed by time slice and singles bin index.
 Array<2, int> _singles;
 
 std::vector<double> _times;
 std::vector<int> _total_prompts;
 std::vector<int> _total_randoms;


#ifdef HAVE_LLN_MATRIX
 Main_header _singles_main_header;
#endif

 int _num_time_slices;

 // A value of zero for _singles_time_interval indicates that the time slices
 // are of different lengths.
 double _singles_time_interval;
//PW change this to BLF filename
 std::string _listmode_filename;

 // Calculate and set _singles_time_interval.
 void set_time_interval();

 // get slice start time.
 double get_slice_start(int slice_index) const;
 

 virtual void set_defaults();
 virtual void initialise_keymap();
 virtual bool post_processing();
 
};








END_NAMESPACE_STIR


#endif
