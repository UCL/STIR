//
// $Id$
//
/*!

  \file
  \ingroup ECAT
  \ingroup IO
  \brief Declaration of class ECAT7OutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000-2$Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __stir_IO_ECAT7OutputFileFormat_H__
#define __stir_IO_ECAT7OutputFileFormat_H__

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
  Implementation of OutputFileFormat paradigm for the ECAT7 format.
 */
class ECAT7OutputFileFormat : 
  public RegisteredParsingObject<
        ECAT7OutputFileFormat,
        OutputFileFormat,
        OutputFileFormat>
{
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  ECAT7OutputFileFormat(const NumericType& = NumericType::SHORT, 
                   const ByteOrder& = ByteOrder::native);

  virtual NumericType set_type_of_numbers(const NumericType&, const bool warn = false);
  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
  //virtual ByteOrder set_byte_order_and_type_of_numbers(ByteOrder&, NumericType&, const bool warn = false);

  virtual Succeeded  
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
