//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class stir::ParsingObject 

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
#ifndef __stir_ParsingObject_H__
#define __stir_ParsingObject_H__


#include "stir/KeyParser.h"

START_NAMESPACE_STIR

/*!
  \ingroup buildblock
  \brief A base class for objects that want to be able to parse parameter files.

  This class is essentially a wrapper for KeyParser, such that it is safe to 
  copy ParsingObject objects. The problem with KeyParser is that it stores 
  pointers to the variables it needs to fill in. So, if you copy one KeyParser
  object to another, both will fill in the same variables (unless add_key is
  called afterwards). ParsingObject solves this by having a copy constructor 
  that reinitialises all keys in its own (protected) KeyParser object.

  \warning All this only works when all keys are set in the initialise_keymap() 
  function, and \b only there.
  \see KeyParser
*/

class ParsingObject 
{
public:
   ParsingObject() ;
   ParsingObject(const ParsingObject&) ;
   ParsingObject& operator=(const ParsingObject&) ;
  virtual ~ParsingObject() {}
  
  /*! \name parsing functions

      parse() returns false if there is some error, true otherwise.
  */
  //@{
   bool parse(istream& f);
   bool parse(const char * const filename);
   //@}

   void ask_parameters();

   string parameter_info();  

protected:
  //! Set defaults before parsing
  virtual void set_defaults();
  //! Initialise all keywords
  virtual void initialise_keymap();
    //! This will be called at the end of the parsing
  /*! \return false if everything OK, true if not */
  virtual bool post_processing(); 

  //! This will be called before parsing or parameter_info is called
  /*! 
    This virtual function should be overloaded when the values for the keywords 
    depend on other variables in the derived class that can be set independently 
    of the parsing.

  \par Example: 

  A derived class has a public member angle_in_radians, while a keyword sets a 
  private member angle_in_degrees.
  */
   virtual void set_key_values();

private:
  bool keymap_is_initialised;  
protected:
  KeyParser parser;

};


END_NAMESPACE_STIR

#endif

