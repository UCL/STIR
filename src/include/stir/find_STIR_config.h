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

//! The following gets the STIR configuration directory 
std::string get_STIR_config_dir();


END_NAMESPACE_STIR
//#include "findSTIRConfig.inl"
#endif // __stir_FINDSTIRCONFIG_H
