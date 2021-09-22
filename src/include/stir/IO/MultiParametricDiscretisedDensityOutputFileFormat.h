//
//
/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
    Copyright (C) 2020 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup MultiIO
  \brief Declaration of class stir::MultiParametricDiscretisedDensityOutputFileFormat

  \author Kris Thielemans

*/

#ifndef __stir_IO_MultiParametricDiscretisedDensityOutputFileFormat_H__
#define __stir_IO_MultiParametricDiscretisedDensityOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"

START_NAMESPACE_STIR

/*!
  \ingroup MultiIO
  \brief 
  Implementation of OutputFileFormat paradigm for the Multi format.
 */
template <typename DiscDensityT>
class MultiParametricDiscretisedDensityOutputFileFormat : 
  public RegisteredParsingObject<
        MultiParametricDiscretisedDensityOutputFileFormat<DiscDensityT>,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> >,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> > >
{
 private:
  typedef 
     RegisteredParsingObject<
        MultiParametricDiscretisedDensityOutputFileFormat<DiscDensityT>,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> >,
        OutputFileFormat<ParametricDiscretisedDensity<DiscDensityT> > >
    base_type;
public :
    //! Name which will be used when parsing an OutputFileFormat object
  static const char * const registered_name;

  MultiParametricDiscretisedDensityOutputFileFormat(const NumericType& = NumericType::FLOAT, 
                   const ByteOrder& = ByteOrder::native);


  virtual ByteOrder set_byte_order(const ByteOrder&, const bool warn = false);
 protected:
  virtual Succeeded  
    actual_write_to_file(std::string& output_filename,
		  const ParametricDiscretisedDensity<DiscDensityT>& density) const;


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  /// Output type for the individual images
  using DiscretisedDensityType = typename ParametricDiscretisedDensity<DiscDensityT>::SingleDiscretisedDensityType::hierarchy_base_type;
  shared_ptr<OutputFileFormat<DiscretisedDensityType> >
    individual_output_type_sptr;
};



END_NAMESPACE_STIR


#endif
