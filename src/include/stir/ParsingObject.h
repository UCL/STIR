//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Declaration of class ParsingObject 

  \author Kris Thielemans
  \author Sanida Mustafovic

  \date $Date$
  \version $Revision$
*/
#ifndef __Tomo_ParsingObject_H__
#define __Tomo_ParsingObject_H__


#include "KeyParser.h"

START_NAMESPACE_TOMO

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
  
  // sm made functions return bool
   bool parse(istream& f);
  //! parse() returns false if there is some error, true otherwise
   bool parse(const char * const filename);
  
   void ask_parameters();

   string parameter_info();  

protected:
  //! Set defaults before parsing
  virtual void set_defaults()  = 0;
  //! Initialise all keywords
  virtual void initialise_keymap() = 0;
    //! This will be called at the end of the parsing
  /*! \return false if everything OK, true if not */
  virtual bool post_processing() 
   { return false; }

  //! This will be called before parsing or parameter_info is called
  /*! 
    This virtual function should be overloaded when the values for the keywords 
    depend on other variables in the derived class that can be set independently 
    of the parsing.

  \par Example: 

  A derived class has a public member angle_in_radians, while a keyword sets a 
  private member angle_in_degrees.
  */
  virtual void set_key_values() {}

private:
  bool keymap_is_initialised;  
protected:
  KeyParser parser;

};


END_NAMESPACE_TOMO

#endif

