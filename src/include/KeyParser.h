//
// $Id$: $Date$
//
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class KeyParser

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$
  \version $Revision$

  \warning All this is going to change ...
*/
#ifndef __KEYPARSER_H__
#define __KEYPARSER_H__

#ifdef OLDDESIGN
#include "pet_common.h"
#endif

#include "Tomography_common.h"

#include <map>
#include <string>
#include <iostream>
#include <vector>

#ifndef TOMO_NO_NAMESPACES
using std::vector;
using std::string;
using std::map;
using std::istream;
#endif

START_NAMESPACE_TOMO

typedef vector<int> IntVect;
typedef vector<unsigned long> UlongVect;
typedef vector<double> DoubleVect;


// KT 20/06/98 new
typedef vector<string> ASCIIlist_type;


/*!
  \ingroup buildblock
  \brief A class to parse Interfile headers
  \warning This class will change dramatically in the future.

  Main problem: this can only be used by derived classes (as it requires
  pointers to member functions.
*/
class KeyParser;

/*!
  \ingroup buildblock
  \brief  A class that enumerates the possible types that can be used
 to store parsed values. Used (only) by KeyParser.

  ASCIIlist means a string which has to be one of a specified list
 */
class KeyArgument
{
public:
  enum type {NONE,ASCII,LIST_OF_ASCII,ASCIIlist, ULONG,INT,
    LIST_OF_INTS,DOUBLE, LIST_OF_DOUBLES, listASCIIlist};
};

/*!
 \brief Class to store the Interfile keywords and their actions.

 \warning These should not be used outside of the KeyParser implementation.
*/
class map_element
{
public :
  KeyArgument::type type;
  void (KeyParser::*p_object_member)();	// pointer to a member function
  //TODO void (*p_object_member)();
  void *p_object_variable;		// pointer to a variable 
  const ASCIIlist_type *p_object_list_of_values;// only used by ASCIIlist

  map_element();

  map_element(KeyArgument::type t, void (KeyParser::*pom)(),
	      void* pov= 0, const ASCIIlist_type *list_of_valid_keywords = 0);

  ~map_element();
	
  map_element& operator=(const map_element& me);
};

typedef map<string,map_element> Keymap;

class KeyParser
{

public:
  KeyParser();
  virtual ~KeyParser();

  //! parse() returns false if there is some error, true otherwise
  bool parse(istream& f);
  //! parse() returns false if there is some error, true otherwise
  bool parse(const char * const filename);

  ////// functions to add keys and their actions 

  typedef void (KeyParser::*KeywordProcessor)();
  // TODO typedef void (*KeywordProcessor)();


  //! add a keyword to the list, together with its call_back function
  void add_key(const string& keyword, 
    KeyArgument::type t, KeywordProcessor function,
    void* variable= 0, const ASCIIlist_type * const list = 0);
  
  //! version that defaults 'function' to set_variable
  void add_key(const string& keyword, KeyArgument::type t, 
	      void* variable, const ASCIIlist_type * const list = 0);


protected : 

  ////// work horses

  //! This will be called at the end of the parsing
  /*! \return false if everything OK, true if not */
  virtual bool post_processing() 
   { return false; }
	


  ////// predefined actions 

  //! to start parsing, has to be set by first keyword
  void start_parsing();
  //! to stop parsing
  void stop_parsing();
  //! for keys which do not do anything
  void do_nothing() {};
  //! set the variable to the value given as the value of the keyword
  void set_variable();



private :

  ////// variables

  enum parse_status { end_parsing, parsing };
  parse_status status;


  Keymap kmap;

  
  istream * input;
  map_element* current;
  int current_index;
  string keyword;

  // could be made into a union
  vector<string>  par_asciilist;
  IntVect	par_intlist;
  DoubleVect	par_doublelist;
  string	par_ascii;
  int		par_int;	
  double	par_double;	
  unsigned long par_ulong;

  ////// methods

  // loops over all lines in the file. returns 0 of OK, 1 of not.
  int parse_header();

  // read a line, see if it's a keyword using map_keyword)
  // if so, return 1, else
  // conditionally write a warning and return 0
  int parse_line(const bool write_warning);

  // set 'current' to map_element corresponding to 'keyword'
  // return 1 if valid keyword, 0 otherwise
  int map_keyword(const string& keyword);

  // call appropriate member function
  void process_key();


  // This function is used by set_variable only
  // It returns the index of a string in 'list_of_values', -1 if not found
  int find_in_ASCIIlist(const string& par_ascii, const ASCIIlist_type& list_of_values);

};

END_NAMESPACE_TOMO

#endif

