//
// $Id$
//
/*!

  \file
  \ingroup IO
  \brief Implementation of class InterfileOutputFileFormat

  \author Kris Thielemans

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002-2$Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "stir/IO/InterfileOutputFileFormat.h"
#include "stir/interfile.h"

START_NAMESPACE_STIR


const char * const 
InterfileOutputFileFormat::registered_name = "Interfile";

InterfileOutputFileFormat::
InterfileOutputFileFormat(const NumericType& type, 
                   const ByteOrder& byte_order) 
{
  type_of_numbers = type;
  file_byte_order = byte_order;
}

void 
InterfileOutputFileFormat::
initialise_keymap()
{
  parser.add_start_key("Interfile Output File Format Parameters");
  parser.add_stop_key("End Interfile Output File Format Parameters");
  OutputFileFormat::initialise_keymap();
}

template <int num_dimensions, typename elemT>
Succeeded  
InterfileOutputFileFormat::
    write_to_file(const string& filename, 
                  const DiscretisedDensity< num_dimensions,elemT>& density)
{
  return 
    write_basic_interfile(filename, desnity);
};



END_NAMESPACE_STIR


