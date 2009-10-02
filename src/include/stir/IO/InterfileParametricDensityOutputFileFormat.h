//
// $Id$
//
/*
    Copyright (C) 2002-$Date$, Hammersmith Imanet Ltd
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
  \brief Declaration of class stir::InterfileParametricDensityOutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#ifndef __stir_IO_InterfileParametricDensityOutputFileFormat_H__
#define __stir_IO_InterfileParametricDensityOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

//template <int num_dimensions, typename elemT> class ParametricDiscretisedDensity;
template <typename DiscDensityT> class ParametricDiscretisedDensity;

/*!
  \ingroup InterfileIO
  \brief 
  Implementation of OutputFileFormat paradigm for the Interfile format.
 */
#if 0
template <int num_dimensions, typename elemT>
class InterfileParametricDensityOutputFileFormat : 
  public RegisteredParsingObject<
        InterfileParametricDensityOutputFileFormat<num_dimensions, elemT>,
        OutputFileFormat<ParametricDiscretisedDensity<num_dimensions, elemT> >,
        OutputFileFormat<ParametricDiscretisedDensity<num_dimensions, elemT> > >
#else
template <typename DiscDensityT>
class InterfileParametricDensityOutputFileFormat : 
  public RegisteredParsingObject<
        InterfileParametricDensityOutputFileFormat<DiscDensityT>,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> >,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> > >
#endif
{
 private:
  typedef 
#if 0
     RegisteredParsingObject<
        InterfileParametricDensityOutputFileFormat<num_dimensions, elemT>,
        OutputFileFormat<ParametricDiscretisedDensity<num_dimensions, elemT> >,
        OutputFileFormat<ParametricDiscretisedDensity<num_dimensions, elemT> > >
#else
     RegisteredParsingObject<
        InterfileParametricDensityOutputFileFormat<DiscDensityT>,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> >,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> > >
#endif
    base_type;
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  InterfileParametricDensityOutputFileFormat(const NumericType& = NumericType::FLOAT, 
                   const ByteOrder& = ByteOrder::native);


  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
 protected:
  virtual Succeeded  
    actual_write_to_file(string& output_filename,
		  const ParametricDiscretisedDensity<DiscDensityT>& density) const;


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

};



END_NAMESPACE_STIR


#endif
