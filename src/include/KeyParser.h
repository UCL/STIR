//
// $Id$ :$Date$
//

#ifndef __KEYPARSER_H__
#define __KEYPARSER_H__

#include <map>
// KT 14/12 changed <String.h> to "pet_string.h"
// KT 09/08/98 forget about pet_string
#include <string>
#include <iostream>

#if !defined(__GNUG__) && !defined (__MSL__)
	using namespace std;
#endif

typedef string String;
//#include "pet_string.h"

#include <vector>

// KT 20/06/98 use this pragma only in VC++
#ifdef _MSC_VER
#pragma warning(disable: 4786)
#endif

#define		STATUS_END_PARSING	0
#define		STATUS_PARSING		1

typedef vector<int> IntVect;
typedef vector<unsigned long> UlongVect;
typedef vector<double> DoubleVect;


// KT 20/06/98 new
typedef vector<String> ASCIIlist_type;

class KeyParser;

class KeyArgument
{
public:
  // KT 20/06/98 changed NUMBERlist to LIST_OF_INTS and (old) ASCIIlist to LIST_OF_ASCII
  // (in Interfile 3.3 ASCIIlist really means a string which has to be in a specified list)
  // KT 01/08/98 change number to int, lnumber to ulong, added double
  // KT 09/10/98 moved in this class to avoid global name space conflicts
  enum type {NONE,ASCII,LIST_OF_ASCII,ASCIIlist, ULONG,INT,
    LIST_OF_INTS,DOUBLE,listASCIIlist};
};

class map_element
{
public :
  KeyArgument::type type;
  void (KeyParser::*p_object_member)();	// pointer to a member function
  void* p_object_variable;		// pointer to a variable 
  const ASCIIlist_type *p_object_list_of_values;			// only used by ASCIIlist

  map_element();
  ~map_element();
	
  // KT 20/06/98 added 4th argument for ASCIIlist
  map_element& operator()(KeyArgument::type t, void (KeyParser::*pom)(),
			  void* pov= NULL, const ASCIIlist_type *list = 0);
  map_element& operator=(const map_element& me);
};

// KT 20/06/98 gcc is up to date now, so we can use the 2 arg version of map<>
#if defined(__MSL__)
typedef map<String,map_element, less<String>, allocator<map_element> > Keymap;
#else
typedef map<String,map_element> Keymap;
#endif

class KeyParser
{
public:
  // KT 19/10/98 removed default constructor as unused at the moment
  //KeyParser();
  // KT 16/10/98 changed  to istream&
  KeyParser(istream& f);
  ~KeyParser();

  bool parse()
  {
    init_keys();
    return (parse_header()==0 && post_processing()==0);
  }

protected : 

  Keymap kmap;

  typedef void (KeyParser::*KeywordProcessor)();

  // Override the next two functions to have any functionality.
  virtual void init_keys() = 0;
  // Returns 0 of OK, 1 of not.
  virtual int post_processing() 
  { return 1; }
	

  void SetEndStatus();
  // KT 09/10/98 added for use keys which don't do anything
  void DoNothing() {};
  void SetVariable();

private :
  // KT 20/06/98 new
  // This function is used by SetVariable only
  // It returns the index of a string in 'list_of_values', -1 of not found
  int find_in_ASCIIlist(const String& par_ascii, const ASCIIlist_type& list_of_values);

  
  // variables
  istream * input;
  int	status;				// parsing 0,end_parsing 1
  map_element* current;
  int current_index;
  String keyword;

  // could be made into a union
  vector<String>  par_asciilist;
  // KT 01/08/98 change number to int, lnumber to ulong, added float
  IntVect		par_intlist;
  String		par_ascii;
  int			par_int;	
  double		par_double;	
  unsigned long par_ulong;

private :

  int ParseLine();
  int MapKeyword(String keyword);
  // KT 19/10/98 removed ChangeStatus as unused
  //void ChangeStatus(int value);
  int ProcessKey();
  // KT 19/10/98 small version of old StartParsing()
  // Returns 0 of OK, 1 of not.
  int parse_header();

};

#endif
