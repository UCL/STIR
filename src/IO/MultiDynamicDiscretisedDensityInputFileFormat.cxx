//
//
/*
  Copyright (C) 2006 - 2007-10-08, Hammersmith Imanet Ltd
  Copyright (C) 2013-01-01 - 2013, Kris Thielemans
  Copyight (C) 2018,2020, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
*/

/*!\file
  \ingroup MultiIO
  \brief Implementation of class stir::MultiDynamicDiscretisedDensityInputFileFormat

  \author Kris Thielemans
  \author Richard Brown

*/

#include "stir/IO/MultiDynamicDiscretisedDensityInputFileFormat.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/NumericType.h"
#include "stir/Succeeded.h"
#include "stir/FilePath.h"
#include "stir/MultipleDataSetHeader.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"

START_NAMESPACE_STIR

bool
MultiDynamicDiscretisedDensityInputFileFormat::actual_can_read(const FileSignature& signature, std::istream&) const
{
  //. todo should check if it's an image
  // checking for "multi :"
  const char* pos_of_colon = strchr(signature.get_signature(), ':');
  if (pos_of_colon == NULL)
    return false;
  std::string keyword(signature.get_signature(), pos_of_colon - signature.get_signature());
  return (standardise_interfile_keyword(keyword) == standardise_interfile_keyword("multi"));
}

std::unique_ptr<MultiDynamicDiscretisedDensityInputFileFormat::data_type>
MultiDynamicDiscretisedDensityInputFileFormat::read_from_file(std::istream&) const
{
  // needs more arguments, so we just give up (TODO?)
  unique_ptr<data_type> ret;
  if (is_null_ptr(ret))
    {
      error("failed to read an Multi image from stream");
    }
  return ret;
}
std::unique_ptr<MultiDynamicDiscretisedDensityInputFileFormat::data_type>
MultiDynamicDiscretisedDensityInputFileFormat::read_from_file(const std::string& filename) const
{
  MultipleDataSetHeader header;
  if (header.parse(filename.c_str()) == false)
    error("MultiDynamicDiscretisedDensity:::read_from_file: Error parsing %s", filename.c_str());
  DynamicDiscretisedDensity* dyn_disc_den_ptr = new DynamicDiscretisedDensity;
  dyn_disc_den_ptr->set_num_densities(header.get_num_data_sets());
  ExamInfo exam_info;
  for (std::size_t i = 1U; i <= header.get_num_data_sets(); ++i)
    {
      unique_ptr<DiscretisedDensity<3, float>> t(DiscretisedDensity<3, float>::read_from_file(header.get_filename(i - 1)));

      // Check that there is time frame information
      if (t->get_exam_info().get_time_frame_definitions().get_num_frames() != 1)
        error(format("The individual components of a dynamic image should contain 1 time frame, but image {} contains {}.",
                     i,
                     t->get_exam_info().get_time_frame_definitions().get_num_frames()));
      double start = t->get_exam_info().get_time_frame_definitions().get_start_time(1);
      double end = t->get_exam_info().get_time_frame_definitions().get_end_time(1);
      // Set some info on the first frame
      if (i == 1)
        {
          exam_info = t->get_exam_info();
          exam_info.time_frame_definitions.set_num_time_frames(header.get_num_data_sets());
          shared_ptr<Scanner> scanner_sptr(Scanner::get_scanner_from_name(exam_info.originating_system));
          dyn_disc_den_ptr->set_scanner(*scanner_sptr);
        }
      exam_info.time_frame_definitions.set_time_frame(i, start, end);
      dyn_disc_den_ptr->set_exam_info(exam_info); // need to update time-frame-info before calling set_density
      dyn_disc_den_ptr->set_density(*t, i);
    }
  // Hard wire some stuff for now (TODO?)
  dyn_disc_den_ptr->set_if_decay_corrected(1.);

  return unique_ptr<data_type>(dyn_disc_den_ptr);
}

END_NAMESPACE_STIR
