#ifndef __stir_FINDSTIRCONFIG_H
#define __stir_FINDSTIRCONFIG_H


#include <string>
#include "stir/common.h"
START_NAMESPACE_STIR

/*!
  \brief find full path of a config file
  \return calls get_STIR_config_dir() and prepends it to \a filename

  Does a brief check if the file can be opened.

  \ingroup ancillary
*/
std::string find_STIR_config_file(const std::string& filename);

/*!
  \brief find string with the (full) path of the directory where looks for STIR configuration files
  \return path name

  First checks an environment variable `STIR_CONFIG_DIR`. If that isn't set, 
  it returns the value of the `STIR_CONFIG_DIR` CMake variable set at build
  time (which has a default location in the installation directory).

  \ingroup ancillary
*/
std::string get_STIR_config_dir();


END_NAMESPACE_STIR

#endif // __stir_FINDSTIRCONFIG_H
