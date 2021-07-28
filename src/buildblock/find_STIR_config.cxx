
#include "stdlib.h"
#include "stir/find_STIR_config.h"
#include "stir/info.h"
#include "stir/config.h"
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR

//! the following returns the path of the configuration file "filename we are looking for
std::string find_STIR_config_file(const std::string& filename){
    
    std::string dir;
    dir = STIR_CONFIG_DIR;
    std::ifstream file(dir+"/"+filename);
    if (file)
        info("Using config file from "+dir);
    else
        error("Could note open "+dir+"/"+filename);
    
    return dir+"/"+filename;

}
//! The following finds the STIR configuration directory and prints it out 
std::string get_STIR_config_dir()
{     
    const char * var = std::getenv("STIR_CONFIG_DIR");
    if (var)
       return var;
    else
       return STIR_CONFIG_DIR;
}

END_NAMESPACE_STIR
