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
  
  
*/


#ifndef __KEYPARSER_H__
#define __KEYPARSER_H__

#ifdef OLDDESIGN
#include "pet_common.h"
#endif

#include "shared_ptr.h"

//#include <map>
#include <list>
#include <utility>
#include <string>
#include <iostream>
#include <vector>

#ifndef TOMO_NO_NAMESPACES
using std::vector;
using std::string;
//using std::map;
using std::list;
using std::pair;
using std::istream;
using std::ostream;
#endif

START_NAMESPACE_TOMO

typedef vector<int> IntVect;
typedef vector<unsigned long> UlongVect;
typedef vector<double> DoubleVect;
typedef vector<string> ASCIIlist_type;

class Object;

/*!
  \ingroup buildblock
  \brief A class to parse Interfile headers

  Currently, Interfile 3.3 parsing rules are hard-coded. 
  
  \warning Vectored keys are treated very dangerously: when a keyword
  is assumed to be vectored, run-time errors occur when it is used without [].
  \warning The use of the [*] index for vectored keys is NOT supported.

  \warning All this is going to change to a more general purpose parser...

  Main problem: when non-trivial callback functions have to be used, you need to do it
  via a derived class (as KeyParser requires  pointers to member functions.)

  \warning As KeyParser::add_key stores pointers to the variables where you want 
  your results, it is somewhat dangerous to copy a KeyParser object. That is, its 
  keymap will still point to the same variables.
*/
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
  enum type {NONE,ASCII,LIST_OF_ASCII,ASCIIlist, ULONG,INT,
    LIST_OF_INTS,DOUBLE, LIST_OF_DOUBLES, listASCIIlist,
    // KT 07/02/2001 new
    PARSINGOBJECT, 
    // KT 20/08/2001 new
    SHARED_PARSINGOBJECT,
    FLOAT, BOOL};
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
  // KT 07/02/2001 new
  // TODO should really not be here, but it works for now
  typedef Object * (Parser)(istream*, const string&);
  Parser* parser;

  map_element();

  map_element(KeyArgument::type t, void (KeyParser::*pom)(),
	      void* pov= 0, const ASCIIlist_type *list_of_valid_keywords = 0);
  // KT 07/02/2001 new
  map_element(void (KeyParser::*pom)(),
	      Object** pov, 
              Parser *);
  // KT 20/08/2001 new
  map_element(void (KeyParser::*pom)(),
	      shared_ptr<Object>* pov, 
              Parser *);
  ~map_element();
	
  map_element& operator=(const map_element& me);
};

class Line;

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

  // KT 07/02/2001 all these are new (up to add_parsing_key)

  //! add a keyword. When parsing, parse its value as a float and put it in *variable_ptr
  void add_key(const string& keyword, float * variable_ptr);
  //! add a keyword. When parsing, parse its value as a double and put it in *variable_ptr
  void add_key(const string& keyword, double * variable_ptr);

  //! add a keyword. When parsing, parse its value as a int and put it in *variable_ptr
  void add_key(const string& keyword, int * variable_ptr);

  //! add a keyword. When parsing, parse its value as a int  and put the bool value in *variable_ptr
  /*! The integer should be 0 or 1, corresponding to false and true resp. */
  void add_key(const string& keyword, bool * variable_ptr);
  
  //! SM 02/05/2001
  void add_key(const string& keyword, vector<double> * variable_ptr);

  //! add a keyword. When parsing, parse its value as a string and put it in *variable_ptr
  /*! The 'value' can contain spaces. */
  void add_key(const string& keyword, string * variable_ptr);

  //! add keyword that has to occur before all others
  /*! Example of such a key: INTERFILE*/
  void add_start_key(const string& keyword);
  //! add a keyword that when encountered, will stop the parsing
  void add_stop_key(const string& keyword);
 
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
      ImageFilter<3,float>* filter_ptr;
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
  void add_parsing_key(const string& keyword, ParsingClass** parsed_object_ptr_ptr)
  {
    add_in_keymap(keyword,     
                  map_element(&KeyParser::set_parsing_object, 
                    reinterpret_cast<Object**>(parsed_object_ptr_ptr), 
                    (map_element::Parser *)(&ParsingClass::read_registered_object))
		    );
  }

  //! add keyword corresponding to an object that will parse the next keys itself
  /*! As above, but with a shared_ptr */
  template <typename ParsingClass>
  void add_parsing_key(const string& keyword, shared_ptr<ParsingClass>* parsed_object_ptr_ptr)
  {
    add_in_keymap(keyword,     
                  map_element(&KeyParser::set_shared_parsing_object, 
                    reinterpret_cast<shared_ptr<Object>*>(parsed_object_ptr_ptr), 
                    (map_element::Parser *)(&ParsingClass::read_registered_object))
		    );
  }

  //! Prints all keywords (in random order) to the stream
  void print_keywords_to_stream(ostream&) const;

  // KT 07/02/2001 new 
  //! Returns a string with keywords and their values
  /*! Keywords are listed in the order they are inserted in the keymap 
      (except for start and stop keys which are listed first and last).
      \warning UNSAFE for 'vectored' keys.
      */
  virtual string parameter_info() const;

  // KT 07/02/2001 new 
  //! Ask interactively for values for all keywords
  /*! Keywords are asked in the order they are inserted in the keymap.

      TODO any consistency checks are currently done by post_processing() at the
      end of the parsing. It should be possible to have checks after every question
      such that it can be repeated.

      \warning UNSAFE for 'vectored' keys.
      */
  virtual void ask_parameters();

  
protected : 

  ////// work horses

  //! This will be called at the end of the parsing
  /*! \return false if everything OK, true if not */
  virtual bool post_processing() 
   { return false; }
	

  
  //! convert 'rough' keyword into a standardised form
  virtual string 
    standardise_keyword(const string& keyword) const;




  typedef void (KeyParser::*KeywordProcessor)();
  // TODO typedef void (*KeywordProcessor)();


  //! add a keyword to the list, together with its call_back function
  /*! This provides a more flexible way to add keys with specific call_backs.
      Can currently only be used by derived classes, as KeywordProcessor has to be a 
      pointer to member function.
      \warning this interface to KeyParser will change in a future release */
  void add_key(const string& keyword, 
    KeyArgument::type t, KeywordProcessor function,
    void* variable= 0, const ASCIIlist_type * const list = 0);
  
  //! version that defaults 'function' to set_variable
  /*! \warning this interface to KeyParser will change in a future release */
  void add_key(const string& keyword, KeyArgument::type t, 
	      void* variable, const ASCIIlist_type * const list = 0);

  ////// predefined call_back functions


public:
  //! to start parsing, has to be set by first keyword
  void start_parsing();
  //! to stop parsing
  void stop_parsing();
  //! for keys which do not do anything
  void do_nothing() {};
  //! set the variable to the value given as the value of the keyword
  void set_variable();

  // KT 07/02/2001 new
  //! set the variable by calling the parser (as stored by add_parsing_key()), with argument the value of the keyword
  void set_parsing_object();
  // KT 20/08/2001 new
  //! set the shared_ptr variable by calling the parser (as stored by add_parsing_key()), with argument the value of the keyword
  void set_shared_parsing_object();

  
private :

  friend class map_element;
  ////// variables

  enum parse_status { end_parsing, parsing };
  parse_status status;

  // KT 01/05/2001 changed to a list, to preserve order of keywords
  //typedef map<string,map_element> Keymap;
  typedef list<pair<string, map_element> > Keymap;

  Keymap kmap;
  // KT 01/05/2001 new functions to allow a list type
  map_element* find_in_keymap(const string& keyword);
  void add_in_keymap(const string& keyword, const map_element& new_element);

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

  // read a line, find keyword and call parse_value_in_line()
  int read_and_parse_line(const bool write_warning);

  // see if keyword is in the keymap using map_keyword
  // if so, call its call_back function and return 1, else
  // conditionally write a warning and return 0  
  int parse_value_in_line(Line& line, const bool write_warning);

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

