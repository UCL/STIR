//
// $Id$: $Date$
//

#ifndef __KEYPARSER_H__
#define __KEYPARSER_H__

#include <map>
#include <string>
#include <iostream>
#include <vector>

#include "pet_common.h"

#ifdef _MSC_VER
// disable warning about truncation of identifiers in debugging information
#pragma warning(disable: 4786)
#endif

typedef vector<int> IntVect;
typedef vector<unsigned long> UlongVect;
typedef vector<double> DoubleVect;


// KT 20/06/98 new
typedef vector<string> ASCIIlist_type;

class KeyParser;

/******************************************************************
 A class that enumerates the possible types that can be used
 to store parsed values.
 ******************************************************************/
class KeyArgument
{
public:
  // KT 20/06/98 changed NUMBERlist to LIST_OF_INTS and (old) ASCIIlist to LIST_OF_ASCII
  // (in Interfile 3.3 ASCIIlist really means a string which has to be in a specified list)
  // KT 01/08/98 change number to int, lnumber to ulong, added double
  // KT 09/10/98 moved in this class to avoid global name space conflicts
  // KT 29/10/98 added LIST_OF_DOUBLES
  enum type {NONE,ASCII,LIST_OF_ASCII,ASCIIlist, ULONG,INT,
    LIST_OF_INTS,DOUBLE, LIST_OF_DOUBLES, listASCIIlist};
};

/******************************************************************
 Classes to store the keywords and their actions.
 These should not be used outside of the KeyParser implementation.
 Unfortunately C++ seems to require defining this class before the 
 class KeyParser.
 ******************************************************************/
class map_element
{
public :
  KeyArgument::type type;
  void (KeyParser::*p_object_member)();	// pointer to a member function
  //TODO void (*p_object_member)();
  void *p_object_variable;		// pointer to a variable 
  const ASCIIlist_type *p_object_list_of_values;// only used by ASCIIlist

  map_element();
  // KT 22/10/98 changed operator() into constructor
  map_element(KeyArgument::type t, void (KeyParser::*pom)(),
	      void* pov= 0, const ASCIIlist_type *list = 0);

  ~map_element();
	
  map_element& operator=(const map_element& me);
};

#if defined(__MSL__)
typedef map<string,map_element, less<string>, allocator<map_element> > Keymap;
#else
typedef map<string,map_element> Keymap;
#endif

/******************************************************************
 KT 26/10/98 modified text
 KeyParser is the main class. It is mainly useful to derive from.

 TODO: write how it works
 ******************************************************************/
class KeyParser
{

public:
  // KT 13/11/98 moved istream arg to parse()
  KeyParser();
  ~KeyParser();

  // parse() returns false if there is some error, true otherwise
  bool parse(istream& f);
  // KT 13/11/98 new
  bool parse(const char * const filename);

  // KT 29/10/98 made next ones public
  ////// functions to add keys and their actions 

  typedef void (KeyParser::*KeywordProcessor)();
  // TODO typedef void (*KeywordProcessor)();

  void add_key(const string& keyword, 
    KeyArgument::type t, KeywordProcessor function,
    void* variable= 0, const ASCIIlist_type * const list = 0);
  
  // version that defaults 'function' to set_variable
  void add_key(const string& keyword, KeyArgument::type t, 
	      void* variable, const ASCIIlist_type * const list = 0);


protected : 

  ////// work horses

  //KT 26/10/98 removed virtual void init_keys() = 0;
  // Override the next function if you need to
  // Returns 0 of OK, 1 of not.
  virtual int post_processing() 
  { return 1; }
	

  ////// predefined actions 

  // to start parsing, has to be set by first keyword
  void start_parsing();
  // to stop parsing
  void stop_parsing();
  // KT 09/10/98 added for keys which don't do anything
  void do_nothing() {};
  // set the variable to the value given as the value of the keyword
  void set_variable();



private :

  ////// variables

  // KT 22/10/98 new enum
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


  // KT 20/06/98 new
  // This function is used by set_variable only
  // It returns the index of a string in 'list_of_values', -1 if not found
  int find_in_ASCIIlist(const string& par_ascii, const ASCIIlist_type& list_of_values);

};

#endif
