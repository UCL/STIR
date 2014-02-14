//
//
/*
    Copyright (C) 2005- 2005 , Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_AbsTimeIntervalFromECAT7ACF__H__
#define __stir_AbsTimeIntervalFromECAT7ACF__H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeIntervalFromECAT7ACF

  \author Kris Thielemans
*/

#include "stir/RegisteredParsingObject.h"
#include "local/stir/AbsTimeInterval.h"
#include <string>

START_NAMESPACE_STIR
class Succeeded;

/*! \ingroup buildblock

  \brief class for specifying a time interval via an ECAT7 .a file

  The ECAT7 header for a .a file does not record the scan duration, but only
  the start time of the transmission scan. So, we have to explicitly ask for 
  the duration.
  
*/
class AbsTimeIntervalFromECAT7ACF
: public RegisteredParsingObject<AbsTimeIntervalFromECAT7ACF,
				 AbsTimeInterval,
				 AbsTimeInterval>
{

public:
  //! Name which will be used when parsing a AbsTimeInterval object 
  static const char * const registered_name; 

  virtual ~AbsTimeIntervalFromECAT7ACF() {}
  //! default constructor sets duration to -1 (i.e. ill-defined)
  AbsTimeIntervalFromECAT7ACF();
  //! read info from ECAT7 file
  /*! will call error() if something goes wrong */
  AbsTimeIntervalFromECAT7ACF(const std::string& filename, 
			      const double duration_in_secs);
  
 private:
  std::string _attenuation_filename; 
  double _transmission_duration;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  Succeeded set_times();

};

END_NAMESPACE_STIR
#endif
