/*
    Copyright (C) 2017-2019, University of Leeds
    Copyright (C) 2020-2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup singles_buildblock
  \ingroup GE
  \brief Implementation of stir::GE::RDF_HDF5::SinglesRatesFromGEHDF5

  \author Palak Wadhwa
  \author Kris Thielemans
*/

#include "stir/IndexRange.h"
#include "stir/IndexRange2D.h"
#include "stir/data/SinglesRatesFromGEHDF5.h"
#include "stir/stream.h"
#include "stir/IO/GEHDF5Wrapper.h"
#include "stir/error.h"

#include <vector>
#include <fstream>
#include <algorithm>
#include <string>

START_NAMESPACE_STIR
namespace GE
{
namespace RDF_HDF5
{

const char* const SinglesRatesFromGEHDF5::registered_name = "SinglesRatesFromGEHDF5";

// Constructor
SinglesRatesFromGEHDF5::SinglesRatesFromGEHDF5()
{}

void
SinglesRatesFromGEHDF5::read_from_file(const std::string& rdf_filename)
{

  unsigned int slice = 0;

  // PW Open the list mode file here.
  m_input_sptr.reset(new GEHDF5Wrapper(rdf_filename));

  SinglesRates::scanner_sptr = m_input_sptr->get_scanner_sptr();
  // Get total number of bins for this type of scanner.
  const int total_singles_units = SinglesRates::scanner_sptr->get_num_singles_units();

  m_input_sptr->initialise_singles_data();

  // Allocate the main array.
  _num_time_slices = m_input_sptr->get_num_singles_samples();
  // GE uses unsigned int, while SinglesRateForTimeSlices uses int, so we need a copy
  Array<2, unsigned int> GE_singles(IndexRange2D(0, _num_time_slices - 1, 0, total_singles_units - 1));

  while (slice < static_cast<unsigned>(_num_time_slices))
    {
      m_input_sptr->read_singles(GE_singles[slice], slice + 1);
      ++slice;
    }

  // copy them across
  _singles.grow(GE_singles.get_index_range());
  std::copy(GE_singles.begin_all(), GE_singles.end_all(), _singles.begin_all());

  // PW Modify this bit of code too.
  if (slice != static_cast<unsigned>(_num_time_slices))
    {
      error("\nSinglesRatesFromGEHDF5: Couldn't read all records in the file. Read %d of %d. Exiting\n", slice, _num_time_slices);
      // TODO resize singles to return array with new sizes
    }
  _times = std::vector<double>(_num_time_slices);
  if (_num_time_slices > 1) // this is currently the same as checking if the input file is a listmode file
    {
      // GE RDF9 listmode stores singles every second
      _singles_time_interval = 1.0;
      for (unsigned int slice = 0; slice < static_cast<unsigned>(_num_time_slices); ++slice)
        _times[slice] = slice + _singles_time_interval; // note that this has to store the "end-time" of the slice

      assert(_times.size() != 0);
    }
  else // Then it must be a sinogram, and therefore only has 1 time and 1 interval.
    {
      TimeFrameDefinitions tf = m_input_sptr->get_exam_info_sptr()->get_time_frame_definitions();
      _times[0] = tf.get_end_time(1);
      _singles_time_interval = tf.get_duration(1);
    }
}

void
SinglesRatesFromGEHDF5::initialise_keymap()
{
  parser.add_start_key("SinglesRatesFromGEHDF5");
  parser.add_key("filename", &_rdf_filename);
  parser.add_stop_key("End SinglesRatesFromGEHDF5");
}

bool
SinglesRatesFromGEHDF5::post_processing()
{
  read_from_file(_rdf_filename);
  return false;
}

void
SinglesRatesFromGEHDF5::set_defaults()
{
  _rdf_filename = "";
}

} // namespace RDF_HDF5
} // namespace GE
END_NAMESPACE_STIR
