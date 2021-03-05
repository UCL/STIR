
#include <cstdlib>
#include "stir/findSTIRConfig.h"
#include "stir/info.h"
#include <iostream>
#include <fstream>

START_NAMESPACE_STIR

std::string find_STIR_config_file(std::string filename){
    std::string dir= getenv("STIR_CONFIG_DIR");
    std::ifstream file(dir+"/"+filename);
    
    if (file)
        info("Using config file from "+dir);
    else{
        std::string pref=getenv("CMAKE_INSTALL_PREFIX");
        dir=pref+"/share/stir/config";
        info("Using config file from "+dir);
    }
    return dir+"/"+filename;

}
END_NAMESPACE_STIR