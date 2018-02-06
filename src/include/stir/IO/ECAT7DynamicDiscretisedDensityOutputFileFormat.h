//
//
/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
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
  \ingroup ECAT
  \brief Declaration of class stir::ecat::ecat7::ECAT7DynamicDiscretisedDensityOutputFileFormat

  \author Kris Thielemans
  \author Charalampos Tsoumpas

*/

#ifndef __stir_IO_ECAT7DynamicDiscretisedDensityOutputFileFormat_H__
#define __stir_IO_ECAT7DynamicDiscretisedDensityOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"
// include for namespace macros
#include "stir/IO/stir_ecat_common.h"
#include <string>

START_NAMESPACE_STIR

class DynamicDiscretisedDensity;

START_NAMESPACE_ECAT
START_NAMESPACE_ECAT7


/*!
  \ingroup ECAT
  \brief 
  Implementation of OutputFileFormat paradigm for the ECAT7 format.

  \warning Currently output always uses 2-byte signed integers in
  big-endian byte order.
 */

class ECAT7DynamicDiscretisedDensityOutputFileFormat : 
  public RegisteredParsingObject<
        ECAT7DynamicDiscretisedDensityOutputFileFormat,
        OutputFileFormat<DynamicDiscretisedDensity>,
        OutputFileFormat<DynamicDiscretisedDensity> >
{
private:
  typedef
     RegisteredParsingObject<
        ECAT7DynamicDiscretisedDensityOutputFileFormat,
        OutputFileFormat<DynamicDiscretisedDensity>,
        OutputFileFormat<DynamicDiscretisedDensity> >
    base_type;

public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  ECAT7DynamicDiscretisedDensityOutputFileFormat(const NumericType& = NumericType::SHORT, 
					 const ByteOrder& = ByteOrder::native);

  //! Set type of numbers to be used for output
  /*! Currently the return value will always be NumericType::SHORT */
  virtual NumericType set_type_of_numbers(const NumericType&, const bool warn = false);
  //! Set byte order to be used for output
  /*! Currently the return value will always be ByteOrder::BIGENDIAN */
  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
  //virtual ByteOrder set_byte_order_and_type_of_numbers(ByteOrder&, NumericType&, const bool warn = false);
public:
  std::string default_scanner_name;

protected:

  virtual Succeeded  
    actual_write_to_file(std::string& output_filename,
		  const DynamicDiscretisedDensity & density) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};

END_NAMESPACE_ECAT7
END_NAMESPACE_ECAT
END_NAMESPACE_STIR

#endif
