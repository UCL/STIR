#ifndef __stir_FINDSTIRCONFIG_H
#define __stir_FINDSTIRCONFIG_H


#include <string>
#include "stir/common.h"
START_NAMESPACE_STIR

std::string find_STIR_config_file(const std::string& filename);
void find_STIR_config_dir();


END_NAMESPACE_STIR
//#include "findSTIRConfig.inl"
#endif // __stir_FINDSTIRCONFIG_H
