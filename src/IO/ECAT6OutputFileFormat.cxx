//
// $Id$
//
/*!

  \file
  \ingroup IO
  \brief Implementation of class ECAT6OutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002-2$Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/IO/ECAT6OutputFileFormat.h"
#include "stir/NumericType.h"
#include "stir/CTI/cti_utils.h"
#include "stir/CTI/stir_cti.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR


const char * const 
ECAT6OutputFileFormat::registered_name = "ECAT6";

ECAT6OutputFileFormat::
ECAT6OutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  type_of_numbers = type;
  file_byte_order = byte_order;
}

void 
ECAT6OutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("ECAT6 Output File Format Parameters");
  parser.add_stop_key("End ECAT6 Output File Format Parameters");
  parser.add_key("default scanner name", &default_scanner_name);
  OutputFileFormat::initialise_keymap();
}

void 
ECAT6OutputFileFormat::
set_defaults()
{
  default_scanner_name = "ECAT 953";
  OutputFileFormat::set_defaults();
}

bool
ECAT6OutputFileFormat::
post_processing()
{
  if (OutputFileFormat::post_processing())
    return true;

  shared_ptr<Scanner> scanner_ptr = 
    Scanner::get_scanner_from_name(default_scanner_name);

  if (find_CTI_system_type(*scanner_ptr)==0)
    {
      warning("ECAT6OutputFileFormat: default_scanner_name %s is not supported\n",
	      default_scanner_name.c_str());
      return true;
    }

  if (file_byte_order != ByteOrder::little_endian)
  {
    warning("ECAT6OutputFileFormat: byte_order is currently fixed to little-endian\n");
    file_byte_order = ByteOrder::little_endian;
  }
  const NumericType supported_type_of_numbers = 
     NumericType("signed integer", 2);
  if (type_of_numbers != supported_type_of_numbers)
  {
    warning("ECAT6OutputFileFormat: output type of numbers is currently fixed to short (2 byte signed integers)\n");
    type_of_numbers = supported_type_of_numbers;
  }

  return false;
}

Succeeded  
ECAT6OutputFileFormat::
    write_to_file(string& filename, 
                  const DiscretisedDensity<3,float>& density) const
{
  shared_ptr<Scanner> scanner_ptr = 
    Scanner::get_scanner_from_name(default_scanner_name);
  
  add_extension(filename, ".img");

  Main_header mhead;
  make_ECAT6_main_header(mhead, *scanner_ptr, "", density);
  mhead.num_frames = 1;
  
  FILE *fptr= cti_create (filename.c_str(), &mhead);
  if (fptr == NULL)
  {
    warning("ECAT6OutputFileFormat::write_to_file: error opening output file %s\n", 
	    filename.c_str());
    return Succeeded::no;
  }

  Succeeded success =
    DiscretisedDensity_to_ECAT6(fptr,
    density, 
    mhead,
    1 /*frame_num*/);
  fclose(fptr);
  
  return success;
}

END_NAMESPACE_STIR


