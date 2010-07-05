//
// $Id$
//
/*
    Copyright (C) 2005- $Date$ , Hammersmith Imanet Ltd
    For internal GE use only
*/
#ifndef __stir_AbsTimeIntervalWithParsing__H__
#define __stir_AbsTimeIntervalWithParsing__H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeIntervalWithParsing

  \author Kris Thielemans
  $Date$
  $Revision$
*/

#include "stir/RegisteredParsingObject.h"
#include "local/stir/AbsTimeInterval.h"

START_NAMESPACE_STIR
class Succeeded;

/*! \ingroup buildblock

  \brief class for specifying a time interval via parsing of explicit times

  This class really only exists to handle the case of parsing the
  start and end times from a file. This functionality could potentially be put
  in AbsTimeInterval, but that would give an overlap/conflict with keywords of
  other derived classes.
  
*/
class AbsTimeIntervalWithParsing
: public RegisteredParsingObject<AbsTimeIntervalWithParsing,
				 AbsTimeInterval,
				 AbsTimeInterval>
{

public:
  //! Name which will be used when parsing a AbsTimeInterval object 
  static const char * const registered_name; 

  virtual ~AbsTimeIntervalWithParsing() {}
  //! default constructor sets times to invalid values
  AbsTimeIntervalWithParsing();
  
 private:

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_STIR

#endif
