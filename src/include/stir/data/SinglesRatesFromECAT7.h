//
// $Id$
//
/*!
  \file
  \ingroup local_buildblock

  \brief Declaration of class SinglesRatesFromECAT7

  \author  Sanida Mustafovic and Kris Thielemans
  $Date$
  $Revision$ 
*/

/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef HAVE_LLN_MATRIX
#error This file can only be compiled when HAVE_LLN_MATRIX is #defined
#endif

#ifndef __stir_SinglesRatesFromECAT7_H__
#define __stir_SinglesRatesFromECAT7_H__

#include "local/stir/SinglesRates.h"
#include "stir/Array.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/IO/stir_ecat7.h"
#include "local/stir/listmode/TimeFrameDefinitions.h"

START_NAMESPACE_STIR
START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7

class SinglesRatesFromECAT7 : 
public RegisteredParsingObject<SinglesRatesFromECAT7, SinglesRates>

{ 
public:

 struct sgl_str {
	  long int  time;
	  long int  num_sgl;
	  long int  sgl[126];
 };

  //! Name which will be used when parsing a SinglesRatesFromECAT7 object 
  static const char * const registered_name; 

  //! Default constructor 
  SinglesRatesFromECAT7 ();

  //!  The function that reads singles from ECAT7 file
  Array<3,float> read_singles_from_file(const string& ECAT7_filename,
					const ios::openmode open_mode = ios::in);
  //! The function that reads singles from *.sgl file 
  Array<3,float> read_singles_from_sgl_file (const string& filename,
					     TimeFrameDefinitions& time_def);
  
  //! Given the detection position get the singles rate   
  virtual float get_singles_rate (const DetectionPosition<>& det_pos, 
				  const float start_time,
				  const float end_time) const;
  
  
private:
  Array<3,float> singles;
  string ECAT7_filename;
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  //TODO 
  int transBlocksPerBucket;
  int angularCrystalsPerBlock;
  int axialCrystalsPerBlock;
  //TimeFrameDefinitions time_def;

  
};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR


#endif
