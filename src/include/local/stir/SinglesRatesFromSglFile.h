//
// $Id$
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup local_buildblock

  \brief Declaration of class SinglesRatesFromSglFile

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$ 
  $Revision$
*/

#ifndef __stir_SinglesRatesFromSglFile_H__
#define __stir_SinglesRatesFromSglFile_H__

#include "local/stir/SinglesRates.h"
#include "stir/Array.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/IO/stir_ecat7.h"


START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7




class SinglesRatesFromSglFile : 
public RegisteredParsingObject<SinglesRatesFromSglFile, SinglesRates>

{ 
public:

 struct sgl_str 
 {
  long int  time;
  long int  num_sgl;
  long int  sgl[126]; // Total prompts and total randoms at the end.
 };

 static const unsigned SIZE_OF_SINGLES_RECORD;

 //! Name which will be used when parsing a SinglesRatesFromSglFile object 
 static const char * const registered_name; 


 //! Default constructor 
 SinglesRatesFromSglFile ();


 //! Given the detection position get the singles rate   
 virtual float get_singles_rate(const DetectionPosition<>& det_pos, 
                                const double start_time,
                                const double end_time) const;


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
 int get_singles_rate(int singles_bin_index, int time_slice) const;

 //! Set a singles rate by time bin index and time slice.
 void set_singles_rate(int singles_bin_index, int time_slice, int new_rate);


  
 //! Get the vector of time values for each time slice index.
 vector<double> get_times() const;


 // Some inspectors

 //! Return the number of time slices.
 int get_num_time_slices() const;

 
 //! Return the time interval per slice of singles data.
 double get_singles_time_interval() const;
 

 // IO Methods

 //! The function that reads singles from *.sgl file.
 int read_singles_from_sgl_file(const string& sgl_filename);

 /*!
  * \brief Write the SinglesRatesFromSglFile object to a stream.
  * \param[in] output The ostream to which the object will be written.
  */
 std::ostream& write(std::ostream& output);



private:
 
 // Indexed by time slice and singles bin index.
 Array<2, int> _singles;
 
 vector<double> _times;
 vector<int> _total_prompts;
 vector<int> _total_randoms;


#ifdef HAVE_LLN_MATRIX
 Main_header _singles_main_header;
#endif
 
 int _num_time_slices;
 double _singles_time_interval;

 string _sgl_filename;
 
 float get_singles_rate(int singles_bin_index, 
                        double start_time, double end_time) const;

 virtual void set_defaults();
 virtual void initialise_keymap();
 virtual bool post_processing();
 
};








END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif
