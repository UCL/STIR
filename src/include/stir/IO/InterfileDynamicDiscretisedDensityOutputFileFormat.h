//
//
/*
    Copyright (C) 2006-2007, Hammersmith Imanet Ltd
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
  \ingroup InterfileIO
  \brief Declaration of class stir::InterfileDynamicDiscretisedDensityOutputFileFormat

  \author Kris Thielemans

*/

#ifndef __stir_IO_InterfileDynamicDiscretisedDensityOutputFileFormat_H__
#define __stir_IO_InterfileDynamicDiscretisedDensityOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

class DynamicDiscretisedDensity;

/*!
  \ingroup InterfileIO
  \brief
  Implementation of OutputFileFormat paradigm for the Interfile format.
 */

class InterfileDynamicDiscretisedDensityOutputFileFormat
    : public RegisteredParsingObject<InterfileDynamicDiscretisedDensityOutputFileFormat,
                                     OutputFileFormat<DynamicDiscretisedDensity>, OutputFileFormat<DynamicDiscretisedDensity>> {
private:
  typedef RegisteredParsingObject<InterfileDynamicDiscretisedDensityOutputFileFormat, OutputFileFormat<DynamicDiscretisedDensity>,
                                  OutputFileFormat<DynamicDiscretisedDensity>>
      base_type;

public:
  //! Name which will be used when parsing an OutputFileFormat object
  static const char* const registered_name;

  InterfileDynamicDiscretisedDensityOutputFileFormat(const NumericType& = NumericType::FLOAT,
                                                     const ByteOrder& = ByteOrder::native);

  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);

protected:
  virtual Succeeded actual_write_to_file(std::string& output_filename, const DynamicDiscretisedDensity& density) const;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
};

END_NAMESPACE_STIR

#endif
