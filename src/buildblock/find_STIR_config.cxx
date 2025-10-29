/*!
  Copyright (C) 2021, National Physical Laboratory
  Copyright (C) 2022, University College London
  This file is part of STIR.

  SPDX-License-Identifier: Apache-2.0

  See STIR/LICENSE.txt for details
  \file
  \ingroup ancillary
  \brief implementation of functions to get configuration directory etc

  \author Daniel Deidda
  \author Kris Thielemans
*/

#include "stir/find_STIR_config.h"
#include "stir/error.h"
#include "stir/config.h"
#include <iostream>
#include <fstream>
#include <cstdlib>

START_NAMESPACE_STIR

std::string
find_STIR_config_file(const std::string& filename)
{

  std::string dir;
  dir = get_STIR_config_dir();
  // TODO this might be dangerous on Windows but seems to work
  const std::string name = (dir + "/" + filename);
  std::ifstream file(name);
  if (!file)
    error("find_STIR_config_file could not open " + name + "\nYou can set the STIR_CONFIG_DIR environment variable to help.");
  if (file.peek() == std::ifstream::traits_type::eof())
    error("find_STIR_config_file error opening file for reading (non-existent or empty file). Filename:\n'" + name + "'");
  return name;
}

std::string
get_STIR_config_dir()
{
  const char* var = std::getenv("STIR_CONFIG_DIR");
  if (var)
    return var;
  else
    return STIR_CONFIG_DIR;
}

std::string
get_STIR_doc_dir()
{
  const char* var = std::getenv("STIR_DOC_DIR");
  if (var)
    return var;
  else
    return STIR_DOC_DIR;
}

std::string
get_STIR_examples_dir()
{
  return get_STIR_doc_dir() + "/examples"; // hopefully works on Windows
}

END_NAMESPACE_STIR
