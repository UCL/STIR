/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2007-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2020, 2021, 2023, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::KeyParser

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project
*/

#ifndef __stir_KEYPARSER_H__
#define __stir_KEYPARSER_H__

#include "stir/shared_ptr.h"
#include "stir/Array.h"
#include "stir/warning.h"
#include "boost/any.hpp"

#include <map>
#include <list>
#include <utility>
#include <string>
#include <iostream>
#include <vector>

START_NAMESPACE_STIR

typedef std::vector<std::string> ASCIIlist_type;

class RegisteredObjectBase;
class Succeeded;

class KeyParser;

/*!
  \ingroup buildblock
  \brief  A class that enumerates the possible types that can be used
 to store parsed values. Used (only) by KeyParser.

  ASCIIlist means a string which has to be one of a specified list

  \todo Should be made a protected enum of KeyParser
 */
class KeyArgument
{
public:
  enum type
  {
    NONE,
    ASCII,
    LIST_OF_ASCII,
    ASCIIlist,
    UINT,
    ULONG,
    LONG,
    INT,
    LIST_OF_INTS,
    DOUBLE,
    LIST_OF_DOUBLES,
    listASCIIlist,
    ARRAY2D_OF_FLOATS,
    ARRAY3D_OF_FLOATS,
    BASICCOORDINATE3D_OF_FLOATS,
    BASICCOORDINATE3D_OF_ARRAY3D_OF_FLOATS,
    PARSINGOBJECT,
    SHARED_PARSINGOBJECT,
    FLOAT,
    BOOL
  };
};

/*!
  \ingroup buildblock
 \brief Class to store the Interfile keywords and their actions.

 \warning These should not be used outside of the KeyParser implementation.
 \todo make private class in KeyParser
*/
class map_element
{
public:
  KeyArgument::type type;
  void (KeyParser::*p_object_member)(); // pointer to a member function
  // TODO void (*p_object_member)();
  void* p_object_variable; // pointer to a variable
  int vectorised_key_level;
  const ASCIIlist_type* p_object_list_of_values; // only used by ASCIIlist
  // TODO should really not be here, but it works for now
  typedef RegisteredObjectBase*(Parser)(std::istream*, const std::string&);
  Parser* parser;

  map_element();

  // map_element(KeyArgument::type t, void (KeyParser::*object_member_ptr)());

  map_element(KeyArgument::type t,
              void (KeyParser::*object_member_ptr)(),
              void* variable_ptr,
              const int vectorised_key_level,
              const ASCIIlist_type* list_of_valid_keywords = 0);
  map_element(void (KeyParser::*pom)(), RegisteredObjectBase** pov, Parser*);
  map_element(void (KeyParser::*pom)(), shared_ptr<RegisteredObjectBase>* pov, Parser*);
  ~map_element();

  map_element& operator=(const map_element& me);
};

// note: next text contains double \\ for correct parsing by doxygen
// you should read them as a single  backslash
/*!
  \ingroup buildblock
  \brief A class to parse Interfile headers

  Currently, Interfile 3.3 parsing rules are hard-coded, with
  some extensions. Note that some functions such as `get_keyword()` are `virtual`
  such that a derived class could use non-Interfile parsing (but this might need
  more work).

  KeyParser reads input line by line and parses each line separately.
  It allows for '\\r' at the end of the line (as in files
  originating in DOS/Windows).
  When the line ends with a backslash '\\', the next line
  will just be appended to the current one. That is, '\\'
  acts as continuation marker.

  Lines can have arbitrary length.

  A special facility is provided for using the value of environment
  variables: after reading the whole line, the text is checked for
  occurences of the form <tt>${some_text}$</tt>. These text strings
  are then replaced by the value of the corresponding environment
  variables (using the std::getenv function).

  \warning For end-of-line continuation, the backslash HAS to be
  the last character. Even spaces after it will stop the 'continuation'.

  \warning The use of the [*] index for vectorised keys is NOT supported.

  Main problem: when non-trivial callback functions have to be used, you need to do it
  via a derived class (as KeyParser requires  pointers to member functions.)

  \warning As KeyParser::add_key stores pointers to the variables where you want
  your results, it is somewhat dangerous to copy a KeyParser object. That is, its
  keymap will still point to the same variables.

  \todo add checking functions in the map, as in
   \code add_key("my key",&my_value, check_positive); \endcode
  \todo add facilities for checking (while parsing) if a keyword was present before the current one
*/
class KeyParser
{
  // historical typedefs
  typedef std::vector<int> IntVect;
  typedef std::vector<double> DoubleVect;

public:
  KeyParser();
  virtual ~KeyParser();

  //! parse() returns false if there is some error, true otherwise
  /*! if \a write_warnings is \c false, warnings about undefined keywords will be supressed.*/
  bool parse(std::istream& f, const bool write_warnings = true);
  //! parse() returns false if there is some error, true otherwise
  /*! if \a write_warnings is \c false, warnings about undefined keywords will be supressed.*/
  bool parse(const char* const filename, const bool write_warnings = true);
  /*! if \a write_warnings is \c false, warnings about undefined keywords will be supressed.*/
  bool parse(const std::string&, const bool write_warnings = true);

  ////// functions to add keys and their actions

  // KT 07/02/2001 all these are new (up to add_parsing_key)

  //! add a keyword. When parsing, parse its value as a float and put it in *variable_ptr
  void add_key(const std::string& keyword, float* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as a float and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<float>* variable_ptr);
  //! add a keyword. When parsing, parse its value as a double and put it in *variable_ptr
  void add_key(const std::string& keyword, double* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as a double and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<double>* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as a list of doubles and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<std::vector<double>>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a int and put it in *variable_ptr
  void add_key(const std::string& keyword, int* variable_ptr);
  //! add a keyword. When parsing, parse its value as a int and put it in *variable_ptr
  void add_key(const std::string& keyword, std::vector<int>* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as a int and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<int>* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as a list of ints and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<std::vector<int>>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a int and put it in *variable_ptr
  void add_key(const std::string& keyword, long int* variable_ptr);

  //! add a keyword. When parsing, parse its value as a int and put it in *variable_ptr
  void add_key(const std::string& keyword, unsigned int* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as an unsigned int and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<unsigned int>* variable);
  //! add a keyword. When parsing, parse its value as an unsigned long and put it in *variable_ptr
  void add_key(const std::string& keyword, unsigned long* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as an unsigned long and put it in \c (*variable_ptr)[current_index]
  void add_vectorised_key(const std::string& keyword, std::vector<unsigned long>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a int  and put the bool value in *variable_ptr
  /*! The integer should be 0 or 1, corresponding to false and true resp. */
  void add_key(const std::string& keyword, bool* variable_ptr);

  //! add a keyword. When parsing, parse its value as a list of doubles and put its value in *variable_ptr
  void add_key(const std::string& keyword, std::vector<double>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a list of comma-separated strings and put its value in *variable_ptr
  void add_key(const std::string& keyword, std::vector<std::string>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a 2d array of floats and put its value in *variable_ptr
  void add_key(const std::string& keyword, Array<2, float>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a 3d array of floats and put its value in *variable_ptr
  void add_key(const std::string& keyword, Array<3, float>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a 3d BasicCoordinate of floats and put its value in *variable_ptr
  void add_key(const std::string& keyword, BasicCoordinate<3, float>* variable_ptr);

  //! add a keyword. When parsing, parse its value as a 3d BasicCoordinate of a 3d array of floats and put its value in
  //! *variable_ptr
  void add_key(const std::string& keyword, BasicCoordinate<3, Array<3, float>>* variable_ptr);

  /*! The 'value' can contain spaces. */
  void add_key(const std::string& keyword, std::string* variable_ptr);
  //! add a vectorised keyword. When parsing, parse its value as a string and put it in \c (*variable_ptr)[current_index]
  /*! The 'value' can contain spaces. */
  void add_vectorised_key(const std::string& keyword, std::vector<std::string>* variable_ptr);
  /*!
    \brief add a keyword. When parsing, its string value is checked against
    a list of strings. The corresponding index is stored in
    *variable_ptr.

    If no match is found, a warning message is issued and *variable_ptr
    is set to -1.
  */
  void add_key(const std::string& keyword, int* variable_ptr, const ASCIIlist_type* const list_of_values);

  //! Add keyword this is just ignored by the parser
  void ignore_key(const std::string& keyword);
  //! add keyword that has to occur before all others
  /*! Example of such a key: INTERFILE*/
  void add_start_key(const std::string& keyword);
  //! add a keyword that when encountered, will stop the parsing
  void add_stop_key(const std::string& keyword);

  //! add keyword corresponding to an object that will parse the next keys itself
  /*!
    When this keyword is encountered during parsing, the parser
    ParsingClass::read_registered_object(name) is called, with 'name' set to
    to the value of the keyword.

    ParsingClass has to be derived from RegisteredObject (strictly speaking it only has
    to be derived from Object, and provide a static function void read_registered_object(string&)).

    Example of usage:
    \code
    struct SomeReconstruction
    {
      SomeReconstruction();
      ImageProcessor<3,float>* filter_ptr;
      double beta;
      KeyParser parser;
    };

    SomeReconstruction::SomeReconstruction()
    {
          beta = 1;
          filter_ptr = 0;
          parser.add_start_key("SomeReconstructionParameters");
          parser.add_key("beta ", &beta);
          parser.add_parsing_key("MAP a la MRP filter type", &filter_ptr);
          parser.add_stop_key("END SomeReconstruction Parameters");
    }
    \endcode

    The .par file then would look as follows
    \verbatim
      SomeReconstructionParameters :=
        beta := 0.001
        MAP a la MRP filter type := Median
        ; keywords here appropriate to a Median ImageFilter
      End SomeReconstruction Parameters:= ;
    \endverbatim
  */
  // definition of next function has to be here to get it to compile with VC (it uses member templates)
  template <typename ParsingClass>
  void add_parsing_key(const std::string& keyword, ParsingClass** parsed_object_ptr_ptr)
  {
    add_in_keymap(keyword,
                  map_element(&KeyParser::set_parsing_object,
                              reinterpret_cast<RegisteredObjectBase**>(parsed_object_ptr_ptr),
                              (map_element::Parser*)(&ParsingClass::read_registered_object)));
  }

  //! add keyword corresponding to an object that will parse the next keys itself
  /*! As above, but with a shared_ptr */
  template <typename ParsingClass>
  void add_parsing_key(const std::string& keyword, shared_ptr<ParsingClass>* parsed_object_ptr_ptr)
  {
    add_in_keymap(keyword,
                  map_element(&KeyParser::set_shared_parsing_object,
                              reinterpret_cast<shared_ptr<RegisteredObjectBase>*>(parsed_object_ptr_ptr),
                              (map_element::Parser*)(&ParsingClass::read_registered_object)));
  }

  //! Removes a key from the kep map
  /*! \return \c true if it was found, \c false otherwise */
  bool remove_key(const std::string& keyword);

  //! Add an alias for keyword
  /*! If \c deprecated_key if \c true, a warning will be written when the alias is encountered.

    \c parameter_info() will use the \c keyword, not the \c alias.
   */
  void add_alias_key(const std::string& keyword, const std::string& alias, bool deprecated_key = true);
  //! Prints all keywords (in random order) to the stream
  void print_keywords_to_stream(std::ostream&) const;

  // KT 07/02/2001 new
  //! Returns a string with keywords and their values
  /*! Keywords are listed in the order they are inserted in the keymap
      (except for start and stop keys which are listed first and last).
  */
  virtual std::string parameter_info() const;

  // KT 07/02/2001 new
  //! Ask interactively for values for all keywords
  /*! Keywords are asked in the order they are inserted in the keymap.

      \todo any consistency checks are currently done by post_processing() at the
      end of the parsing. It should be possible to have checks after every question
      such that it can be repeated.

      \bug breaks with  for 'vectorised' keys.
      */
  virtual void ask_parameters();

  //! returns the index of a string in 'list_of_values', -1 if not found
  /*! Implementation note: this function is non-static because it uses
      standardise_keyword
  */
  int find_in_ASCIIlist(const std::string&, const ASCIIlist_type& list_of_values);

protected:
  ////// work horses

  //! This will be called at the end of the parsing
  /*! \return false if everything OK, true if not
    \todo return Succeeded instead.
    \todo rename to \c post_parsing()
  */
  virtual bool post_processing() { return false; }

  //! convert 'rough' keyword into a standardised form
  /*! Calls standardise_interfile_keyword().
. */
  virtual std::string standardise_keyword(const std::string& keyword) const;

  //! gets a keyword from a string
  /*! Find `:=` or `[`. Note that the returned keyword is not standardised yet.
   */
  virtual std::string get_keyword(const std::string&) const;

  typedef void (KeyParser::*KeywordProcessor)();
  // TODO typedef void (*KeywordProcessor)();

  //! add a keyword to the list, together with its call_back function
  /*! This provides a more flexible way to add keys with specific call_backs.
      Can currently only be used by derived classes, as KeywordProcessor has to be a
      pointer to member function.
      \warning this interface to KeyParser will change in a future release */
  void add_key(const std::string& keyword,
               KeyArgument::type t,
               KeywordProcessor function,
               void* variable = 0,
               const ASCIIlist_type* const list = 0);
  //! add a keyword to the list, together with its call_back function
  /*! This provides a more flexible way to add keys with specific call_backs.
      Can currently only be used by derived classes, as KeywordProcessor has to be a
      pointer to member function.
      \warning this interface to KeyParser will change in a future release */
  void add_key(const std::string& keyword,
               KeyArgument::type t,
               KeywordProcessor function,
               void* variable,
               const int vectorised_key_level,
               const ASCIIlist_type* const list = 0);

  //! version that defaults 'function' to set_variable
  /*! \warning this interface to KeyParser will change in a future release */
  void add_key(const std::string& keyword, KeyArgument::type t, void* variable, const ASCIIlist_type* const list = 0);
  //! version that defaults 'function' to set_variable
  /*! \warning this interface to KeyParser will change in a future release */
  void add_key(const std::string& keyword,
               KeyArgument::type t,
               void* variable,
               const int vectorised_key_level,
               const ASCIIlist_type* const list = 0);

public:
  ////// predefined call_back functions
  //! callback function to start parsing, has to be set by first keyword
  void start_parsing();
  //! to stop parsing
  void stop_parsing();
  //! callback function for keys which do not do anything
  void do_nothing(){};
  //! callback function  that sets the variable to the value given as the value of the keyword
  /*! if the keyword had no value, set_variable will do nothing */
  void set_variable();

  //! callback function that sets the variable by calling the parser (as stored by add_parsing_key()), with argument the value of
  //! the keyword
  /*! if the keyword had no value, set_variable will do nothing */
  void set_parsing_object();
  //! callback function that sets the shared_ptr variable by calling the parser (as stored by add_parsing_key()), with argument
  //! the value of the keyword
  /*! if the keyword had no value, set_variable will do nothing */
  void set_shared_parsing_object();

private:
  friend class map_element;
  ////// variables

  enum parse_status
  {
    end_parsing,
    parsing
  };
  parse_status status;

  // KT 01/05/2001 changed to a list, to preserve order of keywords
  // typedef std::map<std::string,map_element> Keymap;
  typedef std::list<std::pair<std::string, map_element>> Keymap;

  Keymap kmap;

  //! typedef for a map of keyword aliases. Key is alias, value is the "real" keyword
  typedef std::map<std::string, std::string> AliasMap;

  AliasMap alias_map;
  AliasMap deprecated_alias_map;

  // KT 01/05/2001 new functions to allow a list type
  map_element* find_in_keymap(const std::string& keyword);
  void add_in_keymap(const std::string& keyword, const map_element& new_element);

  //! check if keyword is in either alias_map or deprecated_alias_map, and return new value
  std::string resolve_alias(const std::string& keyword) const;

  std::istream* input;
  map_element* current;
  int current_index;
  std::string keyword;

  // next will be false when there's only a keyword on the line
  // maybe should be protected (or even public?). At the moment, this is only used by set_variable().
  // That should cover most cases.
  bool keyword_has_a_value;
  boost::any parameter;

  ////// methods

  //! output the value to a stream, depending on the type
  static void value_to_stream(std::ostream& s, const map_element&);
  //! output the (vectorised) values to a stream, depending on the type
  static void vectorised_value_to_stream(std::ostream& s, const std::string& keyword, const map_element& element);

  //! loops over all lines in the file.
  Succeeded parse_header(const bool write_warnings);

  //! read a line, find keyword and call parse_value_in_line()
  Succeeded read_and_parse_line(const bool write_warning);

  //! find keyword and call its call_back
  // see if current keyword is in the keymap using map_keyword
  // if so, call its call_back function and return Succeeded::yes, else
  // conditionally write a warning and return Succeeded::no
  Succeeded parse_value_in_line(const std::string& line, const bool write_warning);

  //! set 'current' to map_element corresponding to 'keyword'
  // return Succeeded::yes if valid keyword, Succeeded::no otherwise
  Succeeded map_keyword(const std::string& keyword);

  //! call appropriate member function
  void process_key();
};

END_NAMESPACE_STIR

#endif
