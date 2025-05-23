//
//
#ifndef __stir_IO_InterfileImageInputFileFormat_h__
#define __stir_IO_InterfileImageInputFileFormat_h__
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - 2013, Kris Thielemans
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InterfileImageInputFileFormat

  \author Kris Thielemans

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/error.h"
#include "stir/is_null_ptr.h"

START_NAMESPACE_STIR

//! Class for reading images in Interfile file-format.
/*! \ingroup IO

*/
class InterfileImageInputFileFormat :
public InputFileFormat<DiscretisedDensity<3,float> >
{
 public:
  virtual const std::string
    get_name() const
  {  return "Interfile"; }

 protected:
  virtual 
    bool 
    actual_can_read(const FileSignature& signature,
		    std::istream& input) const
  {
    //. todo should check if it's an image
    return is_interfile_signature(signature.get_signature());
  }

  virtual unique_ptr<data_type>
    read_from_file(std::istream& input) const
  {
    unique_ptr<data_type> ret(read_interfile_image(input));
    if (is_null_ptr(ret))
      {
	error("failed to read an Interfile image from stream");
      }
    return ret;
  }
  virtual unique_ptr<data_type>
    read_from_file(const std::string& filename) const
  {
    unique_ptr<data_type> ret(read_interfile_image(filename));
    if (is_null_ptr(ret))
      {
	error("failed to read an Interfile image from file \"%s\"", filename.c_str());
      }
    return ret;
  }
};
END_NAMESPACE_STIR

#endif
