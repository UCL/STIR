/*
    Copyright (C) 2013, Institute for Bioengineering of Catalonia
    Copyright (C) 2018, Commonwealth Scientific and Industrial Research Organisation
                        Australian eHealth Research Centre
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup IO
  \brief Declaration of class stir::ITKOutputFileFormat

  \author Berta Marti Fuster
  \author Ashley Gillman
*/

#ifndef __stir_IO_ITKOutputFileFormat_H__
#define __stir_IO_ITKOutputFileFormat_H__

#include "stir/IO/OutputFileFormat.h"
#include "stir/RegisteredParsingObject.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;

/*!
  \ingroup IO
  \brief  Implementation of OutputFileFormat paradigm using the ITK library for writing.

  ITK (http://www.itk.org) has its own registry of file formats, so the current class
  provides an interface to that code. We translate the STIR data to ITK and then use its
  writing code. This translation is currently incomplete however.

  ITK chooses the file format based on the extension of the filename. This is currently
  different from STIR. Therefore, this class can be used to write Nifti (.nii), Teem (.nhdr),
  MetaIO (.mhdr), etc.

  In this class, we provide a default extension that will be
  appended if a filename without extension is used.

  \par Parameters
  \verbatim
  ITK Output File Format Parameters:=
    default extension:= .nhdr ; current default value
  End ITK Output File Format Parameters:=
  \endverbatim

 */
class ITKOutputFileFormat : public RegisteredParsingObject<ITKOutputFileFormat,
                                                           OutputFileFormat<DiscretisedDensity<3, float>>,
                                                           OutputFileFormat<DiscretisedDensity<3, float>>>
{
private:
  typedef RegisteredParsingObject<ITKOutputFileFormat,
                                  OutputFileFormat<DiscretisedDensity<3, float>>,
                                  OutputFileFormat<DiscretisedDensity<3, float>>>
      base_type;

public:
  //! Name which will be used when parsing an OutputFileFormat object
  static const char* const registered_name;

  //! default extension to use if none present
  /*! This will determine the file format used if passing a filename without extension. */
  std::string default_extension;

  ITKOutputFileFormat(const NumericType& = NumericType::FLOAT, const ByteOrder& = ByteOrder::native);

  ByteOrder set_byte_order(const ByteOrder&, const bool warn = false) override;

protected:
  Succeeded actual_write_to_file(std::string& output_filename, const DiscretisedDensity<3, float>& density) const override;

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;
};

END_NAMESPACE_STIR

#endif
