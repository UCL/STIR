/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-04-30, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012-01-29, Kris Thielemans
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
*/

#include "stir/KeyParser.h"
#include "stir/Succeeded.h"
#include "stir/Object.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/stream.h"
#include "stir/is_null_ptr.h"
#include <boost/format.hpp>
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
using std::vector;
using std::string;
//using std::map;
using std::list;
using std::pair;
using std::istream;
using std::ostream;
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

bool KeyParser::parse(const char * const filename, const bool write_warning)
{
   ifstream hdr_stream(filename);
   if (!hdr_stream)
    { 
      warning("KeyParser::parse: couldn't open file %s\n", filename);
      return false;
    }
   return parse(hdr_stream, write_warning);
}

bool KeyParser::parse(istream& f, const bool write_warning)
{
  // print_keywords_to_stream(cerr);

  input=&f;
  return (parse_header(write_warning)==Succeeded::yes && post_processing()==false);
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
    warning(boost::format("KeyParser: keyword '%s' already registered for parsing, overwriting previous value") % keyword);
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
KeyParser::add_key(const string& keyword, unsigned int * variable)
  {
    add_key(keyword, KeyArgument::UINT, variable);
  }

void
KeyParser::add_key(const string& keyword, long int * variable)
  {
    add_key(keyword, KeyArgument::LONG, variable);
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
KeyParser::add_key(const string& keyword, vector<std::string>* variable)
  {
    add_key(keyword, KeyArgument::LIST_OF_ASCII, variable);
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
KeyParser::add_key(const string& keyword, BasicCoordinate<3,float>* variable)
  {
    add_key(keyword, KeyArgument::BASICCOORDINATE3D_OF_FLOATS, variable);
  }

void
KeyParser::add_key(const string& keyword, BasicCoordinate<3,Array<3,float> >* variable)
  {
    add_key(keyword, KeyArgument::BASICCOORDINATE3D_OF_ARRAY3D_OF_FLOATS, variable);
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

  
Succeeded KeyParser::parse_header(const bool write_warning)
{
    
  if (read_and_parse_line(false)  == Succeeded::yes)	
    process_key();
  if (status != parsing)
  { 
    // something's wrong. We're finding data, but we're not supposed to be
    // parsing. We'll exit with an error, but first write a warning.
    // The warning will say that we miss the "start parsing" keyword (if we can find it in the map)

    // find starting keyword
    string start_keyword;
    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {      
      if (i->second.p_object_member == &KeyParser::start_parsing)
      start_keyword = i->first;
    }
    if (start_keyword.length()>0)
    {
      warning("KeyParser error: required first keyword \"%s\" not found\n",
	      start_keyword.c_str());  
    }
    else
    {
      // there doesn't seem to be a start_parsing keyword, so we cannot include it 
      // in the warning. (it could be a side-effect of another key, so we're 
      // not sure if the map is correct or not)
      warning("KeyParser error: data found, but KeyParser status is \"not parsing\". Keymap possibly incorrect");
    }
    return Succeeded::no; 
  }

  while(status==parsing)
    {
    if(read_and_parse_line(write_warning) == Succeeded::yes)	
	process_key();    
    }
  
  return Succeeded::yes;
  
}	

Succeeded KeyParser::read_and_parse_line(const bool write_warning)
{
  string line;  
  // we keep reading a line until it's either non-empty, or we're at the end of the input
  while (true)
    {
      if (!input->good())
	{
	  warning("KeyParser warning: early EOF or bad file");
	  stop_parsing();
	  return Succeeded::no;
	}
 
      read_line(*input, line);
      // check if only white-space, if not, get out of the loop to continue
      std::size_t pos = line.find_first_not_of(" \t");
      if ( pos != string::npos)
        break;
    }

  // gets keyword
  keyword=standardise_keyword(get_keyword(line));
  return parse_value_in_line(line, write_warning);
}


// functions that get arbitrary type parameters from a string (after '=')
// unfortunately, the string type needs special case as istream::operator>> stops a string at white space
// we also do special things for vectors
// they all return Succeeded::yes when there was  a parameter

template <typename T>
static 
Succeeded
get_param_from_string(T& param, const string& s)
{
  const string::size_type cp=s.find('=',0);
  if(cp==string::npos)
    return Succeeded::no;

  istrstream str(s.c_str()+cp+1);
  str >> param;
  return str.fail() ? Succeeded::no : Succeeded::yes;
}

template <>
Succeeded
get_param_from_string(string& param, const string& s)
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

// special handling for vectors if we desire that no {} means only 1 element
// this is currently only used for the matrix_size keywords in InterfileHeader.

template <typename T>
static
Succeeded
get_vparam_from_string(vector<T>& param, const string& s)
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
      return str.fail() ? Succeeded::no : Succeeded::yes;
    }
  }
  return Succeeded::no;
}

template <typename T>
static
Succeeded
get_vparam_from_string(VectorWithOffset<T>& param, const string& s)
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
      return str.fail() ? Succeeded::no : Succeeded::yes;
    }
  }
  return Succeeded::no;
}

// vectors of strings are also special as we need to split the string up if there are commas
template <>
Succeeded
get_vparam_from_string(vector<string>& param, const string& s)
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

// more template trickery such that we can use boost::any

// this class comes from "Modern C++ Design" by Andrei Alexandrescu
template <class T>
struct Type2Type
{
  typedef T type;
};

template <class T>
static
Succeeded 
get_any_param_from_string(boost::any& parameter, Type2Type<T>,  const string& s)
{
  parameter = T();
  // note: don't use cast to reference as it might break VC 6.0
  return
    get_param_from_string(*boost::any_cast<T>(&parameter), s);
}

template <class T>
static
Succeeded 
get_any_vparam_from_string(boost::any& parameter, Type2Type<T>,  const string& s)
{
  parameter = T();
  // note: don't use cast to reference as it might break VC 6.0
  return
    get_vparam_from_string(*boost::any_cast<T>(&parameter), s);
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
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<std::string>(), line) == Succeeded::yes; 
      break;
    case KeyArgument::INT :
    case KeyArgument::BOOL :
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<int>(), line) == Succeeded::yes; 
      break;
    case KeyArgument::UINT :
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<unsigned int>(), line) == Succeeded::yes; 
      break;
    case KeyArgument::ULONG :
      keyword_has_a_value = 
    get_any_param_from_string(this->parameter, Type2Type<unsigned long>(), line) == Succeeded::yes;
        break;
    case KeyArgument::LONG :
      keyword_has_a_value =
    get_any_param_from_string(this->parameter, Type2Type<long int>(), line) == Succeeded::yes;
      break;
    case KeyArgument::DOUBLE :
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<double>(), line) == Succeeded::yes; 
      break;
    case KeyArgument::FLOAT :
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<float>(), line) == Succeeded::yes; 
      break;
    case KeyArgument::LIST_OF_INTS :
      keyword_has_a_value = 
	get_any_vparam_from_string(this->parameter, Type2Type<std::vector<int> >(), line) == Succeeded::yes; 
      break;
    case KeyArgument::LIST_OF_DOUBLES :
      keyword_has_a_value = 
	get_any_vparam_from_string(this->parameter, Type2Type<std::vector<double> >(), line) == Succeeded::yes; 
      break;
    case KeyArgument::LIST_OF_ASCII :
      // TODO enforce {} by writing get_param_from_string for vector<string>
      keyword_has_a_value = 
	get_any_vparam_from_string(this->parameter, Type2Type<std::vector<std::string> >(), line) == Succeeded::yes; 
      break;
    case KeyArgument::ARRAY2D_OF_FLOATS:
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<Array<2,float> >(), line) == Succeeded::yes; 
      break;
    case KeyArgument::ARRAY3D_OF_FLOATS:
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<Array<3,float> >(), line) == Succeeded::yes; 
      break;
    case KeyArgument::BASICCOORDINATE3D_OF_FLOATS:
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<BasicCoordinate<3,float> >(), line) == Succeeded::yes; 
      break;
    case KeyArgument::BASICCOORDINATE3D_OF_ARRAY3D_OF_FLOATS:
      keyword_has_a_value = 
	get_any_param_from_string(this->parameter, Type2Type<BasicCoordinate<3,Array<3,float> > >(), line) == Succeeded::yes; 
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
  const std::string& par_ascii = *boost::any_cast<std::string>(&this->parameter);
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
    error("KeyParser::SHARED_PARSINGOBJECT can't handle vectored keys yet");
  const std::string& par_ascii = *boost::any_cast<std::string>(&this->parameter);
  reinterpret_cast<shared_ptr<Object> *>(current->p_object_variable)->
    reset((*current->parser)(input, par_ascii));
}

// local function to be used in set_variable below
template <typename T1, typename T2>
void static
assign_to_list(T1& mylist, const T2& value, const int current_index, 
	       const string& keyword)
{
  if(mylist.size() < static_cast<unsigned>(current_index))
    {
      error("KeyParser: the list corresponding to the keyword \"%s\" has to be resized "
	      "to size %d. This means you have a problem in the keyword values.",
	      keyword.c_str(), current_index);
      // mylist.resize(current_index);
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
#define KP_case_assign(KeyArgumentValue, type) \
	case KeyArgumentValue : \
	    *reinterpret_cast<type *>(current->p_object_variable) = \
				      * boost::any_cast<type >(&this->parameter); break

	case KeyArgument::BOOL :
	  {
	    const int par_int = * boost::any_cast<int>(&parameter);
	    if (par_int !=0 && par_int != 1)
	      warning("KeyParser: keyword %s expects a bool value which should be 0 or 1\n"
                      " (actual value is %d). A non-zero value will be assumed to mean 'true'\n",
		      keyword.c_str(), par_int);
	    bool* p_bool=(bool*)current->p_object_variable;	// performs the required casting
	    *p_bool=par_int != 0;
	    break;
	  }
	  KP_case_assign(KeyArgument::INT, int);
	  KP_case_assign(KeyArgument::UINT, unsigned int);
	  KP_case_assign(KeyArgument::ULONG,unsigned long);
      KP_case_assign(KeyArgument::LONG,long);
	  KP_case_assign(KeyArgument::DOUBLE,double);
	  KP_case_assign(KeyArgument::FLOAT,float);
	  KP_case_assign(KeyArgument::ASCII, std::string);

	case KeyArgument::ASCIIlist :
	  {
	    const std::string& par_ascii = *boost::any_cast<std::string>(&this->parameter);
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
	  KP_case_assign(KeyArgument::LIST_OF_INTS, IntVect);
	  KP_case_assign(KeyArgument::LIST_OF_DOUBLES, DoubleVect);
	  // sigh... macro expansion fails of type contain commas.... 
	  // Work-around: use typedefs.
	  typedef Array<2,float> KP_array2d;
	  typedef Array<3,float> KP_array3d;
	  typedef BasicCoordinate<3,float> KP_coord;
	  typedef BasicCoordinate<3,Array<3,float> > KP_coord_array3d;
	  KP_case_assign(KeyArgument::ARRAY2D_OF_FLOATS, KP_array2d);
	  KP_case_assign(KeyArgument::ARRAY3D_OF_FLOATS, KP_array3d);
	  KP_case_assign(KeyArgument::BASICCOORDINATE3D_OF_FLOATS, KP_coord);
	  KP_case_assign(KeyArgument::BASICCOORDINATE3D_OF_ARRAY3D_OF_FLOATS,KP_coord_array3d);
	  KP_case_assign(KeyArgument::LIST_OF_ASCII, std::vector<std::string>);
	default :
	  warning("KeyParser error: unknown type. Implementation error");
	  break;
	}
#undef KP_case_assign
    }
  else	// Sets vector elements using current_index
    {
      switch(current->type)
	{
#define KP_case_assign(KeyArgumentValue, type) \
	case KeyArgumentValue : \
	  assign_to_list(*reinterpret_cast<std::vector<type> *>(current->p_object_variable), \
			 * boost::any_cast<type>(&this->parameter), current_index, keyword); \
	  break

	  KP_case_assign(KeyArgument::INT, int);
	  KP_case_assign(KeyArgument::UINT,unsigned int);
      KP_case_assign(KeyArgument::LONG,long);
	  KP_case_assign(KeyArgument::ULONG,unsigned long);
	  KP_case_assign(KeyArgument::DOUBLE,double);
	  KP_case_assign(KeyArgument::FLOAT,float);
	  KP_case_assign(KeyArgument::ASCII, std::string);
	case KeyArgument::ASCIIlist :
	  {
	    const std::string& par_ascii = *boost::any_cast<std::string>(&this->parameter);
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
	  KP_case_assign(KeyArgument::LIST_OF_INTS, IntVect);
	  KP_case_assign(KeyArgument::LIST_OF_DOUBLES, DoubleVect);
	default :
	  
	  warning("KeyParser error: unknown type. Implementation error");
	  break;
	}
#undef KP_case_assign
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

namespace detail
{
  /* A local helper function, essentially equivalent to operator<<(ostream&, const T& var).
     However, it will insert \ characters in front of end-of-line.
     This is used for types for which operator<< can result in multi-line strings.
     If this is not fixed, it would mean that the output of parameter_info() is
     not immediatelly suitable for parsing back.
     Example: suppose there's a keyword that needs a 2d array. If at parsing we have
     my array:={{1,2},{3}}
     then parameter_info would give
     my_array:={{1,2}
     , {3}
     }
     and this would not follow standard syntax. When using the following function
     in parameter_info(), the output will be
     my_array:={{1,2}\
     , {3}\
     }
  */
  template <class T>
  static void to_stream(ostream& s, const T& var, const char continuation_char = '\\')
  {
    // we will first write everything to a temporary stringstream
    // and then read it back, inserting the backslash
#ifdef BOOST_NO_STRINGSTREAM
    // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
    char str[100000];
    strstream stemp(str, 100000);
#else
    std::stringstream stemp;
#endif
    // write to stemp
    stemp << var;
    
    // now read it back, character by character
    while (true)
      {
	char c;
	stemp.get(c);
	if (!stemp)
	  break;
	if (c == '\n')
	    s << continuation_char; // insert continuation character
	s << c;
      }
  }
}

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
      case KeyArgument::UINT:
        s << *reinterpret_cast<unsigned int*>(i->second.p_object_variable); break;
      case KeyArgument::ULONG:
        s << *reinterpret_cast<unsigned long*>(i->second.p_object_variable); break;
      case KeyArgument::LONG:
        s << *reinterpret_cast<long*>(i->second.p_object_variable); break;
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
          
          if (!is_null_ptr(parsing_object_ptr))
	  {
            //std::cerr << "\nBefore *parsing_object_ptr" << endl;	  
            //std::cerr << "\ntypename *parsing_object_ptr " << typeid(*parsing_object_ptr).name() <<std::endl<<std::endl;
	    s << parsing_object_ptr->get_registered_name() << endl;
	    s << parsing_object_ptr->parameter_info();
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
	detail::to_stream(s, *reinterpret_cast<Array<2,float>*>(i->second.p_object_variable)); break;
      case KeyArgument::ARRAY3D_OF_FLOATS:
	detail::to_stream(s, *reinterpret_cast<Array<3,float>*>(i->second.p_object_variable)); break;
      case KeyArgument::BASICCOORDINATE3D_OF_FLOATS:
	{
	    typedef BasicCoordinate<3,float> type;
	    s << *reinterpret_cast<type*>(i->second.p_object_variable);
	    break;
	}
      case KeyArgument::BASICCOORDINATE3D_OF_ARRAY3D_OF_FLOATS:
	{
	    typedef BasicCoordinate<3,Array<3,float> > type;
	    detail::to_stream(s, *reinterpret_cast<type*>(i->second.p_object_variable));
	    break;
	}
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
