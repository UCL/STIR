//
// $Id$
//
/*!

  \file
  \ingroup IO
  \brief Declaration of class ECAT6OutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000-2$Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_ECAT6OutputFileFormat_H__
#define __stir_IO_ECAT6OutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"
#include <string>

#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR


/*!
  \ingroup IO
  \brief 
  Implementation of OutputFileFormat paradigm for the ECAT6 format.
 */
class ECAT6OutputFileFormat : 
  public RegisteredParsingObject<
        ECAT6OutputFileFormat,
        OutputFileFormat,
        OutputFileFormat>
{
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  ECAT6OutputFileFormat(const NumericType& = NumericType::SHORT, 
                   const ByteOrder& = ByteOrder::native);


  Succeeded  
    write_to_file(string& output_filename,
		  const DiscretisedDensity<3,float>& density) const;
public:
  string default_scanner_name;
private:
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};



END_NAMESPACE_STIR


#endif
