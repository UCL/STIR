//
//
#ifndef __stir_IO_InterfileParametricDiscretisedDensityInputFileFormat_h__
#define __stir_IO_InterfileParametricDiscretisedDensityInputFileFormat_h__
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - 2013, Kris Thielemans
    Copyight (C) 2018, University College London
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup IO
  \brief Declaration of class stir::InterfileParametricDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Richard Brown

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/error.h"
#include "stir/is_null_ptr.h"
#include "stir/format.h"
START_NAMESPACE_STIR

//! Class for reading images in Interfile file-format.
/*! \ingroup IO

*/
template <int num_params>
class InterfileParametricDiscretisedDensityInputFileFormat
    : public InputFileFormat<ParametricDiscretisedDensity<VoxelsOnCartesianGrid<KineticParameters<num_params, float>>>>
{
private:
  typedef InputFileFormat<ParametricDiscretisedDensity<VoxelsOnCartesianGrid<KineticParameters<num_params, float>>>> base_type;

public:
  const std::string get_name() const override { return format("Interfile{}param", num_params); }

  typedef typename base_type::data_type data_type; // <- restores unqualified `data_type` below

protected:
  bool actual_can_read(const FileSignature& signature, std::istream& input) const override
  {
    const std::string sig(signature.get_signature());
    //. todo should check if it's an image
    if (!is_interfile_signature(signature.get_signature()))
      return false;
    // We should check how many parameters are declared in the header.
    // We cannot use the signature, because it is limited to 1024 bytes
    const std::streampos orig_pos = input.tellg();
    const int n = find_num_image_data_types(input);
    input.clear(); // clear EOF before seeking back
    input.seekg(orig_pos);
    return n == num_params;
  }

  unique_ptr<data_type> read_from_file(std::istream&) const override
  {
    // needs more arguments, so we just give up (TODO?)
    unique_ptr<data_type> ret; //(read_interfile_dynamic_image(input));
    if (is_null_ptr(ret))
      {
        error("failed to read an Interfile image from stream");
      }
    return ret;
  }
  unique_ptr<data_type> read_from_file(const std::string& filename) const override
  {
    unique_ptr<data_type> ret(read_interfile_parametric_image<num_params>(filename));
    if (is_null_ptr(ret))
      {
        error("failed to read an Interfile image from file \"%s\"", filename.c_str());
      }
    return ret;
  }

private:
  static int find_num_image_data_types(std::istream& input)
  {
    const std::string key = "number of image data types";
    std::string line;
    while (std::getline(input, line))
      {
        const auto key_pos = line.find(key);
        if (key_pos == std::string::npos)
          continue;
        const auto eq_pos = line.find(":=", key_pos);
        if (eq_pos == std::string::npos)
          continue;
        return std::atoi(line.c_str() + eq_pos + 2);
      }
    return -1;
  }
};
END_NAMESPACE_STIR

#endif
