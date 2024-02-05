//
//
/*
    Copyright (C) 2006-2007, Hammersmith Imanet Ltd
    Copyright (C) 2018 - , University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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

  ByteOrder set_byte_order(const ByteOrder&, const bool warn = false) override;
 protected:
  Succeeded  
    actual_write_to_file(std::string& output_filename,
		  const DynamicDiscretisedDensity& density) const override;


  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
  
  /// Output type for the individual images
  shared_ptr<OutputFileFormat<DiscretisedDensity<3,float> > > individual_output_type_sptr;

};



END_NAMESPACE_STIR


#endif
