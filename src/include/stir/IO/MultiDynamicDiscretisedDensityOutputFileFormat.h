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
  \ingroup MultiIO
  \brief Declaration of class stir::MultiDynamicDiscretisedDensityOutputFileFormat

  \author Kris Thielemans
  \author Richard Brown

*/

#ifndef __stir_IO_MultiDynamicDiscretisedDensityOutputFileFormat_H__
#define __stir_IO_MultiDynamicDiscretisedDensityOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/DiscretisedDensity.h"

START_NAMESPACE_STIR

class DynamicDiscretisedDensity;

/*!
  \ingroup MultiIO
  \brief 
  Implementation of OutputFileFormat paradigm for the Multi format.
 */

class MultiDynamicDiscretisedDensityOutputFileFormat : 
  public RegisteredParsingObject<
        MultiDynamicDiscretisedDensityOutputFileFormat,
        OutputFileFormat<DynamicDiscretisedDensity>,
        OutputFileFormat<DynamicDiscretisedDensity> >
{
 private:
  typedef 
     RegisteredParsingObject<
        MultiDynamicDiscretisedDensityOutputFileFormat,
        OutputFileFormat<DynamicDiscretisedDensity>,
        OutputFileFormat<DynamicDiscretisedDensity> >    base_type;
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  MultiDynamicDiscretisedDensityOutputFileFormat(const NumericType& = NumericType::FLOAT, 
                   const ByteOrder& = ByteOrder::native);

  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
 protected:
  virtual Succeeded  
    actual_write_to_file(std::string& output_filename,
		  const DynamicDiscretisedDensity& density) const;


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  
  /// Output type for the individual images
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > individual_output_type_sptr;

};



END_NAMESPACE_STIR


#endif
