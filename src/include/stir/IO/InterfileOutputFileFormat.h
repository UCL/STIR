//
//
/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup InterfileIO
  \brief Declaration of class stir::InterfileOutputFileFormat

  \author Kris Thielemans

*/

#ifndef __stir_IO_InterfileOutputFileFormat_H__
#define __stir_IO_InterfileOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;

/*!
  \ingroup InterfileIO
  \brief 
  Implementation of OutputFileFormat paradigm for the Interfile format.
 */
class InterfileOutputFileFormat : 
  public RegisteredParsingObject<
        InterfileOutputFileFormat,
        OutputFileFormat<DiscretisedDensity<3,float> >,
        OutputFileFormat<DiscretisedDensity<3,float> > >
{
 private:
  typedef 
     RegisteredParsingObject<
        InterfileOutputFileFormat,
        OutputFileFormat<DiscretisedDensity<3,float> >,
        OutputFileFormat<DiscretisedDensity<3,float> > >
    base_type;
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  InterfileOutputFileFormat(const NumericType& = NumericType::FLOAT, 
                   const ByteOrder& = ByteOrder::native);


  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
 protected:
  virtual Succeeded  
    actual_write_to_file(std::string& output_filename,
		  const DiscretisedDensity<3,float>& density) const;


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};



END_NAMESPACE_STIR


#endif
