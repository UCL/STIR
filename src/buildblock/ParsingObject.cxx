//
// %W%: %E%
//
/*!

  \file

  \brief 

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date %E%
  \version %I%
*/
#include "tomo/ParsingObject.h"
#include <fstream>

#ifndef TOMO_NO_NAMESPACE
using std::ifstream;
#endif

START_NAMESPACE_TOMO

ParsingObject::ParsingObject() 
:
  keymap_is_initialised(false)
{}

  
  

ParsingObject::ParsingObject(const ParsingObject& par)
:
  keymap_is_initialised(false)
  {}


ParsingObject&
ParsingObject::operator =(const ParsingObject& par)
{
  if (&par == this) return *this;
  keymap_is_initialised = false;
  return *this;
}

//void
bool
ParsingObject:: parse(istream& in) 
{ 
  // potentially remove the if() and always call initialise_keymap
  if (!keymap_is_initialised)
  {
    initialise_keymap(); 
    keymap_is_initialised = true;
  }
  set_defaults();
  if (!parser.parse(in))
  {
    warning("Error parsing.\n"); 
    return false;
  }
  else if (post_processing()==true)
    {
      warning("Error post processing keyword values.\n"); 
      return false;
    }
  else
    return true;
}


//void
bool
ParsingObject::parse(const char * const filename)
{
  ifstream hdr_stream(filename);
  if (!hdr_stream)
  { 
    error("ParsingObject::parse: couldn't open file %s\n", filename);
    return false;
  }
  return parse(hdr_stream);
}

void
ParsingObject::ask_parameters()
{
  // potentially remove the if() and always call initialise_keymap
  if (!keymap_is_initialised)
  {
    initialise_keymap(); 
    keymap_is_initialised = true;
  }
  set_defaults();

  parser.ask_parameters();
}

string
ParsingObject::parameter_info() 
{ 
  if (!keymap_is_initialised)
  {
    initialise_keymap(); 
    keymap_is_initialised = true;
  }
  return parser.parameter_info(); 
}

END_NAMESPACE_TOMO