//
// $Id$
//
/*!

  \file
  \ingroup IO
  \brief Declaration of class InterfileOutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000-2$Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_InterfileOutputFileFormat_H__
#define __stir_IO_InterfileOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR


/*!
  \ingroup IO
  \brief 
  Implementation of OutputFileFormat paradigm for the Interfile format.
 */
class InterfileOutputFileFormat : 
  public RegisteredParsingObject<
        InterfileOutputFileFormat,
        OutputFileFormat,
        OutputFileFormat>
{
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  InterfileOutputFileFormat(const NumericType& = NumericType::SHORT, 
                   const ByteOrder& = ByteOrder::native);


  Succeeded  
    write_to_file(const string& filename, 
                  const DiscretisedDensity<3,float>& density);

private:
  void initialise_keymap();

};



END_NAMESPACE_STIR


#endif
