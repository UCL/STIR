/*
    Copyright 2026 University College London

    SPDX-License-Identifier: Apache-2.0
    See STIR/LICENSE.txt for details
 */
/*!

  \file
  \ingroup IO
  \brief Utility function(s) for HDF5 files.

  \author Kris Thielemans
*/

#ifndef __stir_IO_HDF5_utils_H__
#define __stir_IO_HDF5_utils_H__

#include "stir/common.h"
#include <string>
#include <utility>
#include "H5Cpp.h"
#include "H5Exception.h"

START_NAMESPACE_STIR

/*!
   \defgroup HDF5_utilities
   \ingroup IO
   @{
 */

//! Various utility functions for easier HDF5 file processing
namespace HDF5_utilities
{
typedef std::pair<H5E_auto2_t, void*> HDF5_error_handler_t;

inline HDF5_error_handler_t
HDF5_turn_off_error_handling()
{
  HDF5_error_handler_t h;
  H5::Exception::getAutoPrint(h.first, &h.second);
  H5::Exception::dontPrint();
  return h;
}

inline void
HDF5_set_error_handler(HDF5_error_handler_t& h)
{
  H5::Exception::setAutoPrint(h.first, h.second);
}

inline bool
HDF5_dataset_exists(const H5::H5File& file, const std::string& dataset_name)
{
  auto h = HDF5_turn_off_error_handling();
  const auto ret = H5Lexists(file.getId(), dataset_name.c_str(), H5P_DEFAULT);
  HDF5_set_error_handler(h);
  return ret > 0;
}

inline bool
HDF5_group_exists(const H5::H5File& file, const std::string& group_name)
{
  // call same C function
  return HDF5_dataset_exists(file, group_name);
}

inline H5::Group
get_HDF5_group(const H5::H5File& file, const std::string& group_name)
{
  if (!HDF5_group_exists(file, group_name))
    {
      H5::Group group;
      return group;
    }
  return file.openGroup(group_name);
}

inline bool
is_valid(const H5::H5Object& obj)
{
  return obj.getId() != H5I_INVALID_HID;
}

} // namespace HDF5_utilities

/*! @} End of Doxygen Group*/

END_NAMESPACE_STIR

#endif
