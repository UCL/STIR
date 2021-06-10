
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
//! The following finds the STIR configuration director and prints it out 
void find_STIR_config_dir(){
    
    std::string dir;
    dir = STIR_CONFIG_DIR;
        info("Config directory is "+dir);
}

END_NAMESPACE_STIR