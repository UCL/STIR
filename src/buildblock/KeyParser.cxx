//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
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

  \brief Implementations for class stir::KeyParser

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/

#include "stir/KeyParser.h"
#include "stir/Succeeded.h"
#include "stir/Object.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/stream.h"
#include <typeinfo>
#include <fstream>
#include <cstring>
#include <cstdlib>
# ifdef BOOST_NO_STDC_NAMESPACE
 namespace std { using ::getenv; }
# endif

#include <strstream>

#ifndef BOOST_NO_STRINGSTREAM
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::cerr;
using std::cout;
using std::cin;
using std::endl;
using std::istrstream;
using std::ostrstream;
#endif

START_NAMESPACE_STIR

/* function that reads a line upto \n

   It allows for \r at the end of the line (as in files
   originating in DOS/Windows).
   When the line ends with continuation_char, the next line
   will just be appended to the current one.

   The continuation_char HAS to be the last character. Even
   spaces after it will stop the 'continuation'.

   This function should be moved somewhere else as it
   might be useful to someone else as well. (TODO)
*/
static void read_line(istream& input, string& line, 
		      const char continuation_char = '\\')
{
  line.resize(0);
  if (!input)
    return;

  string thisline;
  while(true)
    {
#ifndef _MSC_VER
      std::getline(input, thisline);
#else
      /* VC 6.0 getline does not work properly when input==cin.
         It only returns after a 2nd CR is entered. (The entered input is 
         used for the next getline, so that the input is indeed ok. Problem is
         if that in a sequence
            getline(cin,...);
            cout << "prompt";
            getline(cin,...);
         the user sees the 'prompt' only after the 2nd CR.
         So, we replace getline(stream,string) with our own mess...
      */
      {
        const size_t buf_size=512; // arbitrary number here. we'll check if the line was too long below
        char buf[buf_size];
        thisline.resize(0);
        bool more_chars = false;
        do
        {
          buf[0]='\0';        
          input.getline(buf,buf_size);
          thisline += buf;
          if (input.fail() && !input.bad())
          { 
            // either no characters (end-of-line somehow) or buf_size-1
            input.clear();
            more_chars = strlen(buf)==buf_size-1;        
          }  
          else
            more_chars = false;
        }  
        while (more_chars);
      }
#endif
      // check if last character is \r, 
      // in case this is a DOS file, but not a DOS/Windows host
      if (thisline.size() != 0)
	{
	  string::size_type position_of_last_char = 
	    thisline.size()-1;
	  if (thisline[position_of_last_char] == '\r')
	    thisline.erase(position_of_last_char, 1);
	}
      // TODO handle the case of a Mac file on a non-Mac host (EOL on Mac is \r)

      line += thisline;

      // check for continuation
      if (line.size() != 0)
	{
	  string::size_type position_of_last_char = 
	    line.size()-1;
	  if (line[position_of_last_char] == continuation_char)
	    {
	      line.erase(position_of_last_char, 1);
	      // now the while loop will keep on reading
	    }
	  else
	    {
	      // exit the loop
	      break;
	    }
	}
      else
	{
	  // exit the loop
	  break;
	}
    }

  // replace ${text} with value of environment variable
  {
    string::size_type start_of_env_string;
    while ((start_of_env_string = line.find("${")) != string::npos)
      {
	const string::size_type end_of_env_string = line.find('}',start_of_env_string+2);
	if (end_of_env_string == string::npos)
	  break;
	const string::size_type size_of_env_string =
	  end_of_env_string-start_of_env_string+1;
	const string name_of_env_variable = line.substr(start_of_env_string+2, size_of_env_string-3);
	const char * const value_of_env_variable = std::getenv(name_of_env_variable.c_str());
	if (value_of_env_variable == 0)
	  {
	    warning("KeyParser: environment variable '%s' not found. Replaced by empty string.\n"
		    "This happened while parsing the following line:\n%s",
		    name_of_env_variable.c_str(),
		    line.c_str());
	    line.erase(start_of_env_string, size_of_env_string);
	  }
	else
	  {
	    line.replace(start_of_env_string, 
			 size_of_env_string,
			 value_of_env_variable);
	  }
      }
  }
}

// map_element implementation;

map_element::map_element()
{
  type=KeyArgument::NONE;
  p_object_member=0;
  p_object_variable=0;
  p_object_list_of_values=0;
}

map_element::map_element(KeyArgument::type t, 
			 KeyParser::KeywordProcessor pom, 
			 void* pov,
			 const ASCIIlist_type *list_of_values)
{
  type=t;
  p_object_member=pom;
  p_object_variable=pov;
  p_object_list_of_values=list_of_values;
}

map_element::map_element(void (KeyParser::*pom)(),
	      Object** pov, 
              Parser* parser)
  :
  type(KeyArgument::PARSINGOBJECT),
  p_object_member(pom),
  p_object_variable(pov),
  parser(parser)//static_cast<Parser *>(parser))
  {}

map_element::map_element(void (KeyParser::*pom)(),
	      shared_ptr<Object>* pov, 
              Parser* parser)
  :
  type(KeyArgument::SHARED_PARSINGOBJECT),
  p_object_member(pom),
  p_object_variable(pov),
  parser(parser)//static_cast<Parser *>(parser))
  {}

map_element::~map_element()
{
}


map_element& map_element::operator=(const map_element& me)
{
  type=me.type;
  p_object_member=me.p_object_member;
  p_object_variable=me.p_object_variable;
  p_object_list_of_values=me.p_object_list_of_values;
  parser = me.parser;
  return *this;
}

// KeyParser implementation


KeyParser::KeyParser()
{
  
  current_index=-1;
  status=end_parsing;
  current=0;//KTnew map_element();
}

KeyParser::~KeyParser()
{
}

bool KeyParser::parse(const char * const filename)
{
   ifstream hdr_stream(filename);
   if (!hdr_stream)
    { 
      warning("KeyParser::parse: couldn't open file %s\n", filename);
      return false;
    }
    return parse(hdr_stream);
}

bool KeyParser::parse(istream& f)
{
  // print_keywords_to_stream(cerr);

  input=&f;
  return (parse_header()==Succeeded::yes && post_processing()==false);
}


// KT 10/07/2000 new function
/*!
  This follows Interfile 3.3 conventions:
  <ul>
  <li> The characters \c space, \c tab, \c underscore, \c ! are all
       treated as white space and ignored.       
  <li> Case is ignored.
  </ul>
  Note: in this implementation 'ignoring' white space means 'trimming'
  at the start and end of the keyword, and replacing repeated white space
  with a single space.

*/
string 
KeyParser::standardise_keyword(const string& keyword) const
{
  return standardise_interfile_keyword(keyword);
}

// KT 07/10/2002 moved here from Line
string 
KeyParser::get_keyword(const string& line) const
{
  // keyword stops at either := or an index []	
  // TODO should check that = follows : to allow keywords with colons in there
  const string::size_type eok = line.find_first_of(":[",0);
  return line.substr(0,eok);
}

map_element* KeyParser::find_in_keymap(const string& keyword)
{
  for (Keymap::iterator iter = kmap.begin();
       iter != kmap.end();
       ++iter)
  {
     if (iter->first == keyword)
       return &(iter->second);
  }
  // it wasn't there
  return 0;
}

void 
KeyParser::add_in_keymap(const string& keyword, const map_element& new_element)
{
  const string standardised_keyword = standardise_keyword(keyword);
  map_element * elem_ptr = find_in_keymap(standardised_keyword);
  if (elem_ptr != 0)
  {
    warning("KeyParser: overwriting value of the keyword %s\n", keyword.c_str());
    *elem_ptr = new_element;
  }
  else
    kmap.push_back(pair<string,map_element>(standardised_keyword, new_element));
}

void
KeyParser::add_key(const string& keyword, float * variable)
  {
    add_key(keyword, KeyArgument::FLOAT, variable);
  }
  
void
KeyParser::add_key(const string& keyword, double * variable)
  {
    add_key(keyword, KeyArgument::DOUBLE, variable);
  }

void
KeyParser::add_key(const string& keyword, int * variable)
  {
    add_key(keyword, KeyArgument::INT, variable);
  }

void
KeyParser::add_key(const string& keyword, unsigned long * variable)
  {
    add_key(keyword, KeyArgument::ULONG, variable);
  }

void
KeyParser::add_key(const string& keyword, bool * variable)
  {
    add_key(keyword, KeyArgument::BOOL, variable);
  }

void
KeyParser::add_key(const string& keyword, vector<double>* variable)
  {
    add_key(keyword, KeyArgument::LIST_OF_DOUBLES, variable);
  }

void
KeyParser::add_key(const string& keyword, Array<2,float>* variable)
  {
    add_key(keyword, KeyArgument::ARRAY2D_OF_FLOATS, variable);
  }

void
KeyParser::add_key(const string& keyword, Array<3,float>* variable)
  {
    add_key(keyword, KeyArgument::ARRAY3D_OF_FLOATS, variable);
  }

void
KeyParser::add_key(const string& keyword, string * variable)
  {
    add_key(keyword, KeyArgument::ASCII, variable);
  }

void
KeyParser::add_key(const string& keyword, int * variable,
                   const ASCIIlist_type * list_of_values_ptr)
  {
    add_key(keyword, KeyArgument::ASCIIlist, variable, list_of_values_ptr);
  }
 
void
KeyParser::add_start_key(const string& keyword)
  {
    add_key(keyword, KeyArgument::NONE, &KeyParser::start_parsing);
  }
void
KeyParser::add_stop_key(const string& keyword)
  {
    add_key(keyword, KeyArgument::NONE, &KeyParser::stop_parsing);
  }


  

void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			KeywordProcessor function,
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  add_in_keymap(keyword, map_element(t, function, variable, list_of_values));
}

void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  add_in_keymap(keyword, map_element(t, &KeyParser::set_variable, variable, list_of_values));
}

void
KeyParser::print_keywords_to_stream(ostream& out) const
{
  for (Keymap::const_iterator key = kmap.begin(); key != kmap.end(); ++key)
  {
    out << key->first << '\n';
  }
  out << endl;
}

  
Succeeded KeyParser::parse_header()
{
    
  if (read_and_parse_line(false)  == Succeeded::yes)	
    process_key();
  if (status != parsing)
  { 
    // find starting keyword
    string start_keyword;
    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {      
      if (i->second.p_object_member == &KeyParser::start_parsing)
      start_keyword = i->first;
    }
    warning("KeyParser error: required first keyword \"%s\" not found\n",
        start_keyword.c_str());  
    return Succeeded::no; 
  }

  while(status==parsing)
    {
    if(read_and_parse_line(true) == Succeeded::yes)	
	process_key();    
    }
  
  return Succeeded::yes;
  
}	

// KT 13/03/2001 split parse_line() into 2 functions
Succeeded KeyParser::read_and_parse_line(const bool write_warning)
{
  if (!input->good())
  {
    warning("KeyParser warning: early EOF or bad file");
    stop_parsing();
    return Succeeded::no;
  }
 
  string line;  
  read_line(*input, line);

  // gets keyword
  keyword=standardise_keyword(get_keyword(line));
  return parse_value_in_line(line, write_warning);
}


// functions that get arbitrary type parameters from a string (after '=')
// unfortunately, the string type needs special case are istream::operator>> stops a string at white space
// they all return Succeeded::ok when there was  a parameter

template <typename T>
static int get_param_from_string(T& param, const string& s)
{
  const string::size_type cp=s.find('=',0);
  if(cp==string::npos)
    return Succeeded::no;

  istrstream str(s.c_str()+cp+1);
  str >> param;
  return str.fail() ? Succeeded::no : Succeeded::yes;
}

template <>
static int get_param_from_string(string& param, const string& s)
{
  
  const string::size_type cp = s.find('=',0);
  if(cp!=string::npos)
  {
    // skip starting white space
    const string::size_type sok=s.find_first_not_of(" \t",cp+1); // KT 07/10/2002 now also skips tabs
    if(sok!=string::npos)
    {
      // strip trailing white space
      const string::size_type eok=s.find_last_not_of(" \t",s.length());
      param=s.substr(sok,eok-sok+1);
      return Succeeded::yes;
    }
  }
  return Succeeded::no;
}

// sadly, VC 6.0 can't resolve get_param_from_string with and without vector template, so I have to call this function differently
template <typename T>
static int get_vparam_from_string(vector<T>& param, const string& s)
{
  const string::size_type cp = s.find('=',0);
  if(cp!=string::npos)
  {
    // skip starting white space
    const string::size_type start=s.find_first_not_of(" \t",cp+1); // KT 07/10/2002 now also skips tabs
    if(start!=string::npos)
    {
      istrstream str(s.c_str()+start);
      
      if (s[start] == '{')
        str >> param;
      else
      {
        param.resize(1);
        str >> param[0];
      }
      return Succeeded::yes;
    }
  }
  return Succeeded::no;
}

template <typename T>
static int get_vparam_from_string(VectorWithOffset<T>& param, const string& s)
{
  const string::size_type cp = s.find('=',0);
  if(cp!=string::npos)
  {
    // skip starting white space
    const string::size_type start=s.find_first_not_of(" \t",cp+1); // KT 07/10/2002 now also skips tabs
    if(start!=string::npos)
    {
      istrstream str(s.c_str()+start);
      
      if (s[start] == '{')
        str >> param;
      else
      {
        param = VectorWithOffset<T>(); // TODO will NOT work with multi-dimensional arrays
	param.grow(0,0);
        str >> param[0];
      }
      return Succeeded::yes;
    }
  }
  return Succeeded::no;
}

template <>
static int get_vparam_from_string(vector<string>& param, const string& s)
{
  string::size_type cp = s.find('=',0);
  if(cp!=string::npos)
  {
    // skip starting white space
    const string::size_type start=s.find_first_not_of(" \t",cp+1); // KT 07/10/2002 now also skips tabs
    if(start!=string::npos)
    {
      if (s[start] == '{')
      {
        bool end=false;
        cp = start+1;
        while (!end)
        {
          cp=s.find_first_not_of("},",cp);
          cp=s.find_first_not_of(" \t",cp);
          
          if(cp==string::npos)
          {
            end=true;
          }
          else
          {
            string::size_type eop=s.find_first_of(",}",cp);
            if(eop==string::npos)
            {
              end=true;
              eop=s.length();
            }
            // trim ending white space
            const string::size_type eop2 = s.find_last_not_of(" \t",eop);
            param.push_back(s.substr(cp,eop2-cp));
            cp=eop+1;
          }
        }
      }
      else
      {
        param.resize(1);        
        param[0] = s.substr(start, s.find_last_not_of(" \t",s.size()));
      }
      return Succeeded::yes;
    }
  }
  return Succeeded::no;
}

// function that finds the current_index. work to do here!
static int get_index(const string& line)
{
  // we take 0 as a default value for the index
  int in=0;
  // make sure that the index is part of the key (i.e. before :=)
  const string::size_type cp=line.find_first_of(":[",0);
  if(cp!=string::npos && line[cp] == '[')
  {
    const string::size_type sok=cp+1;
    const string::size_type eok=line.find_first_of(']',cp);
    // check if closing bracket really there
    if (eok == string::npos)
    {
      // TODO do something more graceful
      warning("Interfile warning: invalid vectored key in line \n'%s'.\n%s",
        line.c_str(), 
        "Assuming this is not a vectored key.");
      return 0;
    }
    in=atoi(line.substr(sok,eok-sok).c_str());
  }
  return in;
}

Succeeded KeyParser::parse_value_in_line(const string& line, const bool write_warning)
{
  // KT 07/10/2002 use return value of get_param to detect if a value was present at all
  current_index=get_index(line);
    
  // maps keyword to appropriate map_element (sets current)
  if(map_keyword(keyword)==Succeeded::yes)
  {
    switch(current->type)	// depending on the par_type, gets the correct value from the line
    {				// and sets the right temporary variable
    case KeyArgument::NONE :
      keyword_has_a_value = false;
      break;
    case KeyArgument::ASCII :
    case KeyArgument::ASCIIlist :
      // KT 07/02/2001 new
    case KeyArgument::PARSINGOBJECT:
    case KeyArgument::SHARED_PARSINGOBJECT:
      keyword_has_a_value = get_param_from_string(par_ascii, line) == Succeeded::yes;
      break;
    case KeyArgument::INT :
    case KeyArgument::BOOL :
      keyword_has_a_value = get_param_from_string(par_int, line) == Succeeded::yes; 
      break;
    case KeyArgument::ULONG :
      keyword_has_a_value = get_param_from_string(par_ulong, line) == Succeeded::yes; 
      break;
    case KeyArgument::DOUBLE :
    case KeyArgument::FLOAT :
      keyword_has_a_value = get_param_from_string(par_double, line) == Succeeded::yes; 
      break;
    case KeyArgument::LIST_OF_INTS :
      par_intlist.clear();
      keyword_has_a_value = get_vparam_from_string(par_intlist, line) == Succeeded::yes; 
      break;
    case KeyArgument::LIST_OF_DOUBLES :
      par_doublelist.clear();
      keyword_has_a_value = get_vparam_from_string(par_doublelist, line) == Succeeded::yes; 
      break;
    case KeyArgument::LIST_OF_ASCII :
      par_asciilist.clear();
      keyword_has_a_value = get_vparam_from_string(par_asciilist, line) == Succeeded::yes; 
      break;
    case KeyArgument::ARRAY2D_OF_FLOATS:
      par_array2d_of_floats = Array<2,float>();
      keyword_has_a_value = get_vparam_from_string(par_array2d_of_floats, line) == Succeeded::yes; 
      break;
    case KeyArgument::ARRAY3D_OF_FLOATS:
      par_array3d_of_floats = Array<3,float>();
      keyword_has_a_value = get_vparam_from_string(par_array3d_of_floats, line) == Succeeded::yes; 
      break;
    default :
      // KT 07/10/2002 now exit with error
      error ("KeyParser internal error: keyword '%s' has unsupported type of parameters\n",
        keyword.c_str());
      return  Succeeded::no; // just a line to avoid compiler warnings
    }
    return Succeeded::yes;
  }

  // skip empty lines and comments
  if (keyword.length() != 0 && keyword[0] != ';' && write_warning)
    warning("KeyParser warning: unrecognized keyword: %s\n", keyword.c_str());

  // do no processing of this key
  return Succeeded::no;
}

void KeyParser::start_parsing()
{
  status=parsing;
}

void KeyParser::stop_parsing()
{
  status=end_parsing;
}

// KT 07/02/2001 new
void KeyParser::set_parsing_object()
{
  // KT 07/10/2002 new
  if (!keyword_has_a_value)
    return;

  // TODO this does not handle the vectored key convention
  
  // current_index is set to 0 when there was no index
  if(current_index!=0)
    error("KeyParser::PARSINGOBJECT can't handle vectored keys yet\n");
  *reinterpret_cast<Object **>(current->p_object_variable) =
    (*current->parser)(input, par_ascii);	    
}


// KT 20/08/2001 new
void KeyParser::set_shared_parsing_object()
{
  // KT 07/10/2002 new
  if (!keyword_has_a_value)
    return;
  
  // TODO this does not handle the vectored key convention
  
  // current_index is set to 0 when there was no index
  if(current_index!=0)
    error("KeyParser::SHARED_PARSINGOBJECT can't handle vectored keys yet\n");
  *reinterpret_cast<shared_ptr<Object> *>(current->p_object_variable) =
    (*current->parser)(input, par_ascii);	    
}

// local function to be used in set_variable below
template <typename T1, typename T2>
void static
assign_to_list(T1& mylist, const T2& value, const int current_index, 
	       const string& keyword)
{
  if(mylist.size() < static_cast<unsigned>(current_index))
    {
      // this is based on a suggestion by Dylan Togane [dtogane@camhpet.on.ca]
      warning("KeyParser: the list corresponding to the keyword \"%s\" has to be resized "
	      "to size %d. This might mean you have a problem in the keyword values.\n",
	      keyword.c_str(), current_index);
      mylist.resize(current_index);
    }
  mylist[current_index-1] = value;
}

void KeyParser::set_variable()
{
  if (!keyword_has_a_value)
    return;

  // TODO this does not handle the vectored key convention
  
  // current_index is set to 0 when there was no index
  if(!current_index)
    {
      switch(current->type)
	{	  
	case KeyArgument::BOOL :
	  {
	    if (par_int !=0 && par_int != 1)
	      warning("KeyParser: keyword %s expects a bool value which should be 0 or 1\n"
                      " (actual value is %d). A non-zero value will be assumed to mean 'true'\n",
		      keyword.c_str(), par_int);
	    bool* p_bool=(bool*)current->p_object_variable;	// performs the required casting
	    *p_bool=par_int != 0;
	    break;
	  }
	case KeyArgument::INT :
	  {
	    int* p_int=(int*)current->p_object_variable;	// performs the required casting
	    *p_int=par_int;
	    break;
	  }
	 
	case KeyArgument::ULONG :
	  {
	    unsigned long* p_ulong=(unsigned long*)current->p_object_variable;	// performs the required casting
	    *p_ulong=par_ulong;
	    break;
	  }
	 
	case KeyArgument::DOUBLE :
	  {
	    double* p_double=(double*)current->p_object_variable;	// performs the required casting
	    *p_double=par_double;
	    break;
	  }
	case KeyArgument::FLOAT :
	  {
	    float* p_float=(float*)current->p_object_variable;	// performs the required casting
	    // TODO check range
	    *p_float=static_cast<float>(par_double);
	    break;
	  }
	case KeyArgument::ASCII :
	  {
	    string* p_string=(string*)current->p_object_variable;	// performs the required casting
	    *p_string=par_ascii;
	    break;
	  }
	  
	case KeyArgument::ASCIIlist :
	  {
	    const int index =
              find_in_ASCIIlist(par_ascii, *(current->p_object_list_of_values));
	    *((int *)current->p_object_variable) = index;
            if (index == -1)
            { 
              // it was not in the list
              // TODO we should use warning() instead
              cerr << "KeyParser warning : value of keyword \""
                   << keyword << "\" is \""
                   << par_ascii
                   << "\"\n\tshould have been one of:";
              for (unsigned int i=0; i<current->p_object_list_of_values->size(); i++)
                cerr << "\n\t" << (*current->p_object_list_of_values)[i];
              cerr << '\n' << endl;
            }
	    break;
	  }
	case KeyArgument::LIST_OF_INTS :
	  {
	    IntVect* p_vectint=(IntVect*)current->p_object_variable;
	    *p_vectint=par_intlist;
	    break;
	  }
	case KeyArgument::LIST_OF_DOUBLES :
	  {
            // KT 07/02/2001 bug corrected: was a straight copy of the INT case above
	    DoubleVect* p_vect=(DoubleVect*)current->p_object_variable;
	    *p_vect=par_doublelist;
	    break;
	  }
        case KeyArgument::ARRAY2D_OF_FLOATS:
	  {
	    *reinterpret_cast<Array<2,float>*>(current->p_object_variable) =
	      par_array2d_of_floats;
	    break;
	  }
	  case KeyArgument::ARRAY3D_OF_FLOATS:
	  {
	    *reinterpret_cast<Array<3,float>*>(current->p_object_variable) =
	      par_array3d_of_floats;
	    break;
	  }
	case KeyArgument::LIST_OF_ASCII :
	  {
	    vector<string>* p_vectstring=(vector<string>*)current->p_object_variable;
	    *p_vectstring=par_asciilist;
	    break;
	  }
	default :
	  warning("KeyParser error: unknown type. Implementation error");
	  break;
	}
    }
  else	// Sets vector elements using current_index
    {
      switch(current->type)
	{
	case KeyArgument::INT :
	  {
	    assign_to_list(*(IntVect*)current->p_object_variable, par_int, current_index, keyword);
	    break;
	  }
	case KeyArgument::ULONG :
	  {
	    assign_to_list(*(UlongVect*)current->p_object_variable, par_ulong, current_index, keyword);
	    break;
	  }
	  
	case KeyArgument::DOUBLE :
	  {
	    assign_to_list(*(DoubleVect*)current->p_object_variable, par_double, current_index, keyword);
	    break;
	  }
	case KeyArgument::ASCII :
	  {
	    assign_to_list(*(vector<string>*)current->p_object_variable, par_ascii, current_index, keyword);
	    break;
	  }
	  
	case KeyArgument::ASCIIlist :
	  {
	    const int index_in_asciilist =
              find_in_ASCIIlist(par_ascii, *(current->p_object_list_of_values));
	    assign_to_list(*(IntVect*)current->p_object_variable, 
			   index_in_asciilist, current_index, keyword);
            if (index_in_asciilist == -1)
            { 
              // it was not in the list
              // TODO we should use warning() instead
              cerr << "KeyParser warning : value of keyword \""
                   << keyword << "\" is \""
                   << par_ascii
                   << "\"\n\tshould have been one of:";
              for (unsigned int i=0; i<current->p_object_list_of_values->size(); i++)
                cerr << "\n\t" << (*current->p_object_list_of_values)[i];
            }
	    break;
	  }
	case KeyArgument::LIST_OF_INTS :
	  {
	    assign_to_list(*(vector<IntVect>*)current->p_object_variable, par_intlist, current_index, keyword);
	    break;
	  }
	case KeyArgument::LIST_OF_DOUBLES :
	  {
	    assign_to_list(*(vector<DoubleVect>*)current->p_object_variable, par_doublelist, current_index, keyword);
	    break;
	  }
	/*	case LIST_OF_ASCII :
		{
		assign_to_list(*(vector<string>*)current->p_object_variable, par_asciilist, current_index, keyword);
		break;
		}*/
	default :
	  
	  warning("KeyParser error: unknown type. Implementation error");
	  break;
	}
    }
}

int KeyParser::find_in_ASCIIlist(const string& par_ascii, const ASCIIlist_type& list_of_values)
{
  {
    // TODO, once we know for sure type of ASCIIlist_type, we could use STL find()
    // TODO it would be more efficient to call standardise_keyword on the 
    // list_of_values in add_key()
    for (unsigned int i=0; i<list_of_values.size(); i++)
      if (standardise_keyword(par_ascii) == standardise_keyword(list_of_values[i]))
	return i;
  }
  return -1;
}  	

Succeeded KeyParser::map_keyword(const string& keyword)
{
  current = find_in_keymap(keyword);
  return (current==0 ? Succeeded::no : Succeeded::yes);
}



void KeyParser::process_key()
{
  // KT 17/05/2001 replaced NULL with 0 to prevent gcc compiler warning
  if(current->p_object_member!=0)
  {
    (this->*(current->p_object_member))();	//calls appropriate member function
  }
}

// KT 07/02/2001 new 
// TODO breaks with vectored keys (as there is no way of finding out if the
// variable is actually a vector of the relevant type
string KeyParser::parameter_info() const
{  
#ifdef BOOST_NO_STRINGSTREAM
    // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
    char str[100000];
    ostrstream s(str, 100000);
#else
    std::ostringstream s;
#endif

    // first find start key
    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {      
      if (i->second.p_object_member == &KeyParser::start_parsing)
      s << i->first << " :=\n";
    }

    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {
     if (i->second.p_object_member == &KeyParser::start_parsing ||
         i->second.p_object_member == &KeyParser::stop_parsing)
         continue;

      s << i->first << " := ";
      switch(i->second.type)
      {
        // TODO will break with vectored keys
      case KeyArgument::DOUBLE:
        s << *reinterpret_cast<double*>(i->second.p_object_variable); break;
      case KeyArgument::FLOAT:
        s << *reinterpret_cast<float*>(i->second.p_object_variable); break;
      case KeyArgument::INT:
        s << *reinterpret_cast<int*>(i->second.p_object_variable); break;
      case KeyArgument::BOOL:
        s << (*reinterpret_cast<bool*>(i->second.p_object_variable) ? 1 : 0); break;
      case KeyArgument::ULONG:
        s << *reinterpret_cast<unsigned long*>(i->second.p_object_variable); break;
      case KeyArgument::NONE:
        break;
      case KeyArgument::ASCII :
	 s << *reinterpret_cast<string*>(i->second.p_object_variable); break;	  	  
      case KeyArgument::ASCIIlist :
        { 
	    const int index = *reinterpret_cast<int*>(i->second.p_object_variable);
            s << (index == -1 ?
                   "UNALLOWED VALUE" :
                   (*i->second.p_object_list_of_values)[index]);
	    break;
	}
      case KeyArgument::PARSINGOBJECT:
        {
          Object* parsing_object_ptr =
            *reinterpret_cast<Object**>(i->second.p_object_variable);
	  if (parsing_object_ptr!=0)
	  {
	    s << parsing_object_ptr->get_registered_name() << endl;
	    s << parsing_object_ptr->parameter_info() << endl;
	  }
	  else
	    s << "None";
          break;	 
        }
      case KeyArgument::SHARED_PARSINGOBJECT:
        {
#if defined(__GNUC__) && __GNUC__ < 3

          s << "COMPILED WITH GNU C++ (prior to version 3.0), CANNOT INSERT VALUE";
#else
          shared_ptr<Object> parsing_object_ptr =
            (*reinterpret_cast<shared_ptr<Object>*>(i->second.p_object_variable));
          
          if (parsing_object_ptr.use_count()!=0)
	  {
            //std::cerr << "\nBefore *parsing_object_ptr" << endl;	  
            //std::cerr << "\ntypename *parsing_object_ptr " << typeid(*parsing_object_ptr).name() <<std::endl<<std::endl;
	    s << parsing_object_ptr->get_registered_name() << endl;
	    s << parsing_object_ptr->parameter_info() << endl;
	  }
	  else
	    s << "None";
#endif
          break;	 
        }

      case KeyArgument::LIST_OF_DOUBLES:
        s << *reinterpret_cast<DoubleVect*>(i->second.p_object_variable); break;	  	  
      case KeyArgument::LIST_OF_INTS:
        s << *reinterpret_cast<IntVect*>(i->second.p_object_variable); break;	  
      case KeyArgument::ARRAY2D_OF_FLOATS:
	s << *reinterpret_cast<Array<2,float>*>(i->second.p_object_variable); break;
      case KeyArgument::ARRAY3D_OF_FLOATS:
	s << *reinterpret_cast<Array<3,float>*>(i->second.p_object_variable); break;
      case KeyArgument::LIST_OF_ASCII:
        s << *reinterpret_cast<vector<string>*>(i->second.p_object_variable); break;	  	  
      default :
	  warning("KeyParser error: unknown type. Implementation error\n");
	  break;

      }
      s << endl;
    }
    // finally, find stop key
    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {      
      if (i->second.p_object_member == &KeyParser::stop_parsing)
      s << i->first << " := \n";
    }

    return s.str();
  }

// KT 13/03/2001 new 
// TODO breaks with vectored keys (as there is no way of finding out if the
// variable is actually a vector of the relevant type
void KeyParser::ask_parameters()
{   
  // This is necessary for set_parsing_object. It will allow the 
  // 'recursive' parser to see it's being called interactively.
  input = 0;
  while(true)
  {
    string line;
    
    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {
      
      if (i->second.p_object_member == &KeyParser::do_nothing ||
          i->second.p_object_member == &KeyParser::start_parsing ||
          i->second.p_object_member == &KeyParser::stop_parsing)
          continue;

      keyword = i->first;

      cout << keyword << " := ";
      {
	read_line(cin, line);
	// prepend ":=" such that parse_value_in_line can work properly
	line.insert(0, ":= ");
      }

      if (parse_value_in_line(line, false) == Succeeded::yes)
        process_key();    
    }
    if (post_processing())    
       cout << " Asking all questions again! (Sorry)\n";
    else
      return;
  }
}
END_NAMESPACE_STIR
