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
//! the following returns the path of the configuration file "filename we are looking for
std::string find_STIR_config_file(const std::string& filename);

//! The following returns a string with the (full) path of the directory where STIR configuration files are found
//! First checks an environment variable `STIR_CONFIG_DIR`. If that isn't set, 
//! it returns the value of the `STIR_CONFIG_DIR` CMake variable set at build time 
//! (with a default location in the installation directory).
std::string get_STIR_config_dir();


END_NAMESPACE_STIR
//#include "findSTIRConfig.inl"
#endif // __stir_FINDSTIRCONFIG_H
