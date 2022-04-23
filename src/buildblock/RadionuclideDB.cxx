/*!

  \file
  \ingroup ancillary
  \brief Implementation of class stir::RadionuclideDB
  
  \author Daniel Deidda
  \author Kris Thielemans
      
*/
/*
    Copyright (C) 2021, National Physical Laboratory
    Copyright (C) 2021, 2022, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

#include "stir/RadionuclideDB.h"
#include "stir/info.h"
#include "stir/round.h"
#include "stir/error.h"
#include "stir/find_STIR_config.h"

START_NAMESPACE_STIR

RadionuclideDB::
RadionuclideDB()
{
#ifdef nlohmann_json_FOUND
  read_from_file(find_STIR_config_file("radionuclide_info.json"));
  this->radionuclide_lookup_table_filename=find_STIR_config_file("radionuclide_names.json");
#endif
}

void
RadionuclideDB::
read_from_file(const std::string& arg)
{
#ifdef nlohmann_json_FOUND
    this->database_filename = arg;
    
    //Read Radionuclide file and set JSON member for DB
    
    std::string s =this->database_filename;
    std::ifstream json_file_stream(s);
    
    if(!json_file_stream)
        error("Could not open Json file!");
  
  //  nlohmann::json radionuclide_json;
    json_file_stream >> this->radionuclide_json;
  //  radionuclide_json.parse(json_file_stream);
  //  
    
    if (radionuclide_json.find("nuclide") == radionuclide_json.end())
    {
      error("RadionuclideDB: No or incorrect JSON radionuclide set (could not find \"nuclide\" in file \""
            + this->database_filename + "\")");
    }
#else
    error("RadionuclideDB: STIR was compiled without JSON support and therefore cannot read a database.");
#endif
}

std::string 
RadionuclideDB::
get_radionuclide_name_from_lookup_table(const std::string& rname) const
{
  if (rname.empty())
    return "default";

#ifdef nlohmann_json_FOUND
  if (this->radionuclide_lookup_table_filename.empty())
    error("RadionuclideDB: no filename set for look-up table");
    
  std::ifstream json_file_stream(this->radionuclide_lookup_table_filename);
    
  if (!json_file_stream)
    error("Could not open radionuclide lookup file:'" + this->radionuclide_lookup_table_filename + "'");
    
  nlohmann::json table_json;
  json_file_stream >> table_json;
    
//    Check that lookup table and database have the same number of elements
    if (radionuclide_json["nuclide"].size() != table_json.size())
        error("The lookup table and the radionuclide database do not have the same number of elements. " 
              "If you added a radionuclide you also need to add the same in the lookup table");
    
    for (unsigned int l=0; l<table_json.size(); l++)
        for (unsigned int c=0; c<table_json.at(l).size(); c++)
        {
            if(table_json.at(l).at(c)==rname)
            return table_json.at(l).at(0);
        }
    /* not found in table, so return as-is */
    return rname;
#else
    return rname;
#endif
}

Radionuclide
RadionuclideDB::
get_radionuclide_from_json(ImagingModality rmodality, const std::string &rname) const
{
    
  if (this->database_filename.empty())
    error("RadionuclideDB: no filename set for the Radionuclide info");

  std::string name = rname;

  float  keV ;
  float  h_life ;
  float branching_ratio;
  
  info("RadionuclideDB: finding record radionuclide: " + rname+
       " in file "+ this->database_filename);
#ifdef nlohmann_json_FOUND
  //Extract appropriate chunk of JSON file for given nuclide.
  //nlohmann::json target = radionuclide_json["nuclide"][name]["modality"][rmodality.get_name()]["properties"];
  std::string modality_string;
  switch (rmodality.get_modality())
    {
    case ImagingModality::PT:
      modality_string = "PET"; break;
    case ImagingModality::NM:
      modality_string = "nucmed"; break;
    default:
      error("RadionuclideDB::get_radionuclide_from_json called with unknown modality");
    }

  auto rnuclide_entry = radionuclide_json["nuclide"].find(name);
  if (rnuclide_entry == radionuclide_json["nuclide"].end())
    {
      error("RadionuclideDB: radionuclide " + rname + " not found in JSON database");
    }
  auto rnuclide_entry2 = (*rnuclide_entry)["modality"].find(modality_string);
  if (rnuclide_entry2 == (*rnuclide_entry)["modality"].end())
    {
      error("RadionuclideDB: radionuclide " + rname + " modality " + modality_string + " not found in JSON database");
    }
  auto rnuclide_entry3 = rnuclide_entry2->find("properties");
  if (rnuclide_entry3 == rnuclide_entry2->end())
    {
      error("RadionuclideDB: radionuclide " + rname + " modality " + modality_string + " found but properties not in JSON database");
    }
  auto& target = *rnuclide_entry3;

  int location = -1;
  int pos = 0;
  for (auto entry : target){
      location = pos;
    pos++;
  }

  if (location == -1){
    error("RadionuclideDB: Desired radionuclide not found!");
  }

  //Extract properties for specific nuclide and modality.
  nlohmann::json properties = target[location];
  {
    std::stringstream str;
    str << properties.dump(6);
    info("RadionuclideDB: JSON record found:" + str.str(),2);
  }
  try
    {
      keV = properties["keV"];
    }
  catch (...)
    {
      error("RadionuclideDB: energy not set for " + rname);
    }
  try
    {
  branching_ratio = properties["branching_ratio"];
    }
  catch (...)
    {
      error("RadionuclideDB: branching_ratio not set for " + rname);
    }
  try
    {
      h_life = properties["half_life"];
    }
  catch (...)
    {
      error("RadionuclideDB: half_life not set for " + rname);
    }
  
  Radionuclide rnuclide(rname,
                        keV,
                        branching_ratio,
                        h_life,
                        rmodality);
  
  return rnuclide;
#else
  error("Internal error: RadioNuclideDB::get_radionuclide_from_json should never be called when JSON support is not enabled.");
#endif
}

Radionuclide 
RadionuclideDB::
get_radionuclide(ImagingModality rmodality, const std::string& rname)
{
  // handle default case
  if (rname.empty() || rname == "default")
    {
      if (rmodality.get_modality()==ImagingModality::PT)
        return get_radionuclide(rmodality, "^18^Fluorine");
      else if (rmodality.get_modality()==ImagingModality::NM)
        return get_radionuclide(rmodality, "^99m^Technetium");
      else
        error("RadioNuclideDB::get_radionuclide: unknown modality");
    }

  std::string nuclide_name = get_radionuclide_name_from_lookup_table(rname);

#ifdef nlohmann_json_FOUND

  return get_radionuclide_from_json(rmodality,nuclide_name);

#else

    if(rmodality.get_modality()==ImagingModality::PT){
        if (rname != "^18^Fluorine")
          error("RadioNuclideDB::get_radionuclide: since STIR was compiled without nlohmann-json-dev, We only have information for ^18^Fluorine for the PET modality.");

        return Radionuclide("^18^Fluorine",
                              511,
                              0.9686,
                              6584.04,
                              rmodality);
    }else if(rmodality.get_modality()==ImagingModality::NM){
        if (rname != "^99m^Technetium")
          error("RadioNuclideDB::get_radionuclide: since STIR was compiled without nlohmann-json-dev, We only have information for ^99m^Technetium for the NM modality.");

        return Radionuclide("^99m^Technetium",
                              140.511,
                              0.885,
                              21624.12,
                              rmodality);
    }
    else
      error("RadioNuclideDB::get_radionuclide: unknown modality");

#endif
}

END_NAMESPACE_STIR

