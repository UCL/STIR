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
  long int  sgl[126];
 };
 static const unsigned size_of_singles_record;

  //! Name which will be used when parsing a SinglesRatesFromSglFile object 
  static const char * const registered_name; 

  //! Default constructor 
  SinglesRatesFromSglFile ();

  //! The function that reads singles from *.sgl file 
  Array<3,float> read_singles_from_sgl_file (const string& sgl_filename);
  
  //! Given the detection position get the singles rate   
  virtual float get_singles_rate (const DetectionPosition<>& det_pos, 
				  const double start_time,
				  const double end_time) const;
  
  vector<double> get_times() const;

private:
  Array<3,float> singles;
  // TODO move to Scanner
  int num_axial_blocks_per_singles_unit;
  //Array<1,float> times;
  vector<double> times;
  string sgl_filename;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

#ifdef HAVE_LLN_MATRIX
  Main_header singles_main_header;
#endif
  //TODO 
  int trans_blocks_per_bucket;
  int angular_crystals_per_block;
  int axial_crystals_per_block;
  double singles_time_interval;
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif
