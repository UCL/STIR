
#include "stir/find_STIR_config.h"
#include "stir/info.h"
#include "stir/config.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

START_NAMESPACE_STIR

//! the following returns the path of the configuration file "filename we are looking for
std::string find_STIR_config_file(const std::string& filename){
    
    std::string dir;
    dir = get_STIR_config_dir(); //STIR_CONFIG_DIR;
    // TODO this might be dangerous on Windows but seems to work
    const std::string name = (dir+"/"+filename);
    std::ifstream file(name);
    if (!file)
       error("find_STIR_config_file could not open "+name);
    if (file.peek() == std::ifstream::traits_type::eof())
      error("find_STIR_config_file error opening file for reading (non-existent or empty file). Filename:\n'" + name + "'");
    return name;

}
//! The following gets the STIR configuration directory 
std::string get_STIR_config_dir()
{     
    const char * var = std::getenv("STIR_CONFIG_DIR");
    if (var)
       return var;
    else
       return STIR_CONFIG_DIR;
}

END_NAMESPACE_STIR
