//
//
/*
    Copyright (C) 2005- 2010 , Hammersmith Imanet Ltd
    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
*/
#ifndef __stir_AbsTimeIntervalWithParsing__H__
#define __stir_AbsTimeIntervalWithParsing__H__
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::AbsTimeIntervalWithParsing

  \author Kris Thielemans
*/

#include "stir/RegisteredParsingObject.h"
#include "stir_experimental/AbsTimeInterval.h"

START_NAMESPACE_STIR
class Succeeded;

/*! \ingroup buildblock

  \brief class for specifying a time interval via parsing of explicit times

  This class really only exists to handle the case of parsing the
  start and end times from a file. This functionality could potentially be put
  in AbsTimeInterval, but that would give an overlap/conflict with keywords of
  other derived classes.

*/
class AbsTimeIntervalWithParsing : public RegisteredParsingObject<AbsTimeIntervalWithParsing, AbsTimeInterval, AbsTimeInterval>
{

public:
  //! Name which will be used when parsing a AbsTimeInterval object
  static const char* const registered_name;

  ~AbsTimeIntervalWithParsing() override {}
  //! default constructor sets times to invalid values
  AbsTimeIntervalWithParsing();

private:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

#endif
