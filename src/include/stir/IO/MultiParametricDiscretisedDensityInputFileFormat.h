//
//
#ifndef __stir_IO_MultiParametricDiscretisedDensityInputFileFormat_h__
#define __stir_IO_MultiParametricDiscretisedDensityInputFileFormat_h__
/*
    Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013-01-01 - 2013, Kris Thielemans
    Copyight (C) 2018,2020, University College London
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
  \ingroup IO
  \brief Declaration of class stir::MultiParametricDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Richard Brown

*/
#include "stir/IO/InputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/utilities.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/error.h"
#include "stir/is_null_ptr.h"
#include "stir/MultipleDataSetHeader.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include <boost/format.hpp>

START_NAMESPACE_STIR

//! Class for reading images in Multi file-format.
/*! \ingroup IO

*/
class MultiParametricDiscretisedDensityInputFileFormat : public InputFileFormat<ParametricVoxelsOnCartesianGrid> {
public:
  virtual const std::string get_name() const { return "Multi"; }

protected:
  virtual bool actual_can_read(const FileSignature& signature, std::istream&) const {
    //. todo should check if it's an image
    // checking for "multi :"
    const char* pos_of_colon = strchr(signature.get_signature(), ':');
    if (pos_of_colon == NULL)
      return false;
    std::string keyword(signature.get_signature(), pos_of_colon - signature.get_signature());
    return (standardise_interfile_keyword(keyword) == standardise_interfile_keyword("multi"));
  }

  virtual unique_ptr<data_type> read_from_file(std::istream&) const {
    // needs more arguments, so we just give up (TODO?)
    unique_ptr<data_type> ret;
    if (is_null_ptr(ret)) {
      error("failed to read a Multi image from stream");
    }
    return ret;
  }
  virtual unique_ptr<data_type> read_from_file(const std::string& filename) const {
    MultipleDataSetHeader header;
    if (header.parse(filename.c_str()) == false)
      error("MultiParametricDiscretisedDensity:::read_from_file: Error parsing \"" + filename + '\"');
    if (header.get_num_data_sets() != ParametricVoxelsOnCartesianGrid::get_num_params())
      error("MultiParametricDiscretisedDensity:::read_from_file: "
            "wrong number of images found in \"" +
            filename +
            "\"\n"
            "(expected " +
            std::to_string(ParametricVoxelsOnCartesianGrid::get_num_params()) + ")");

    using SingleDiscretisedDensityType = ParametricVoxelsOnCartesianGrid::SingleDiscretisedDensityType;
    auto t = stir::read_from_file<SingleDiscretisedDensityType>(header.get_filename(0));
    ParametricVoxelsOnCartesianGrid* par_disc_den_ptr = new ParametricVoxelsOnCartesianGrid(*t);
    par_disc_den_ptr->update_parametric_image(*t, 1);
    for (std::size_t i = 2; i <= header.get_num_data_sets(); ++i) {
      t = stir::read_from_file<SingleDiscretisedDensityType>(header.get_filename(i - 1));
      // TODO should check all time-frame info consistent
      // exam_info.get_time_frame_definitions TODO.set_time_frame(i, start, end);
      par_disc_den_ptr->update_parametric_image(*t, i);
    }

    return unique_ptr<data_type>(par_disc_den_ptr);
  }
};
END_NAMESPACE_STIR

#endif
