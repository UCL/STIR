//
// $Id$
//
/*!

  \file
  \ingroup InterfileIO
  \brief Declaration of class InterfileOutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2003-$Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_InterfileOutputFileFormat_H__
#define __stir_IO_InterfileOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR


/*!
  \ingroup InterfileIO
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

  InterfileOutputFileFormat(const NumericType& = NumericType::FLOAT, 
                   const ByteOrder& = ByteOrder::native);


  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
 protected:
  virtual Succeeded  
    actual_write_to_file(string& output_filename,
		  const DiscretisedDensity<3,float>& density) const;


  virtual void initialise_keymap();
  virtual bool post_processing();

};



END_NAMESPACE_STIR


#endif
