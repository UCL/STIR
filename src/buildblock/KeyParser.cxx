//
// $Id$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class KeyParser

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/KeyParser.h"
#include "stir/Object.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/line.h"
#include "stir/stream.h"
#include <typeinfo>
#include <fstream>
#include <cstring>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::cerr;
using std::cout;
using std::cin;
using std::endl;
using std::ends;
#endif

START_NAMESPACE_STIR

//! max length of Interfile line
const int MAX_LINE_LENGTH=512;

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
  current=new map_element();
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
  return (parse_header()==0 && post_processing()==false);
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
KeyParser::add_key(const string& keyword, string * variable)
  {
    add_key(keyword, KeyArgument::ASCII, variable);
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
  //kmap[standardise_keyword(keyword)] = 
  // map_element(t, function, variable, list_of_values);
  add_in_keymap(keyword, map_element(t, function, variable, list_of_values));
}

void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  //kmap[standardise_keyword(keyword)] = 
  //  map_element(t, &KeyParser::set_variable, variable, list_of_values);
  add_in_keymap(keyword, map_element(t, &KeyParser::set_variable, variable, list_of_values));
}

// sadly can't print values at the moment
// reason is that that there is currently no information if this is
// is a vectored key or not.
void
KeyParser::print_keywords_to_stream(ostream& out) const
{
  for (Keymap::const_iterator key = kmap.begin(); key != kmap.end(); ++key)
  {
    out << key->first << '\n';
  }
  out << endl;
}

  
int KeyParser::parse_header()
{
    
  if (read_and_parse_line(false))	
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
    return 1; 
  }

  while(status==parsing)
    {
      if(read_and_parse_line(true))	
	process_key();    
    }
  
  return 0;
  
}	

// KT 13/03/2001 split parse_line() into 2 functions
int KeyParser::read_and_parse_line(const bool write_warning)
{
  Line line;
  
  if (!input->good())
  {
    warning("KeyParser warning: early EOF or bad file");
    stop_parsing();
    return 0;
  }
  {
    char buf[MAX_LINE_LENGTH];
    input->getline(buf,MAX_LINE_LENGTH,'\n');
    // check if last character is \r, 
    // in case this is a DOS file, but not a DOS/Windows host
    if (strlen(buf)>0)
    {
      char * last_char = buf+strlen(buf) - 1;
      if (*last_char == '\r')
        *last_char = '\0';
    }
    // TODO handle the case of a Mac file on a non-Mac host (EOL on Mac is \r)

    line=buf;
  }
		// gets keyword
  keyword=standardise_keyword(line.get_keyword());
  return parse_value_in_line(line, write_warning);
}

int KeyParser::parse_value_in_line(Line& line, const bool write_warning)
{
  current_index=line.get_index();
		// maps keyword to appropriate map_element (sets current)
  if(map_keyword(keyword))	
  {
    switch(current->type)	// depending on the par_type, gets the correct value from the line
    {				// and sets the right temporary variable
    case KeyArgument::NONE :
      break;
    case KeyArgument::ASCII :
    case KeyArgument::ASCIIlist :
      // KT 07/02/2001 new
    case KeyArgument::PARSINGOBJECT:
    case KeyArgument::SHARED_PARSINGOBJECT:
      line.get_param(par_ascii);
      break;
    case KeyArgument::INT :
    case KeyArgument::BOOL :
      line.get_param(par_int);
      break;
    case KeyArgument::ULONG :
      line.get_param(par_ulong);
      break;
    case KeyArgument::DOUBLE :
    case KeyArgument::FLOAT :
      line.get_param(par_double);
      break;
    case KeyArgument::LIST_OF_INTS :
      par_intlist.clear();
      line.get_param(par_intlist);
      break;
    case KeyArgument::LIST_OF_DOUBLES :
      par_doublelist.clear();
      line.get_param(par_doublelist);
      break;
    case KeyArgument::LIST_OF_ASCII :
      par_asciilist.clear();
      line.get_param(par_asciilist);
      break;
    default :
      break;
    }
    return 1;
  }

  // skip empty lines and comments
  if (keyword.length() != 0 && keyword[0] != ';' && write_warning)
    warning("KeyParser warning: unrecognized keyword: %s\n", keyword.c_str());

  // do no processing of this key
  return 0;
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
    // TODO this does not handle the vectored key convention
  
  // current_index is set to 0 when there was no index
  if(current_index!=0)
    error("KeyParser::SHARED_PARSINGOBJECT can't handle vectored keys yet\n");
  *reinterpret_cast<shared_ptr<Object> *>(current->p_object_variable) =
    (*current->parser)(input, par_ascii);	    
}

void KeyParser::set_variable()
{
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
	    *((int *)current->p_object_variable) =
	      find_in_ASCIIlist(par_ascii, *(current->p_object_list_of_values));
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
  else														// Sets vector elements using current_index
    {
      switch(current->type)
	{
	case KeyArgument::INT :
	  {
	    IntVect* p_vnumber=(IntVect*)current->p_object_variable;	// performs the required casting
	    p_vnumber->operator[](current_index-1)=par_int;
	    break;
	  }
	case KeyArgument::ULONG :
	  {
	    UlongVect* p_vnumber=(UlongVect*)current->p_object_variable;	// performs the required casting
	    p_vnumber->operator[](current_index-1)=par_ulong;
	    break;
	  }
	  
	case KeyArgument::DOUBLE :
	  {
	    DoubleVect* p_vnumber=(DoubleVect*)current->p_object_variable;	// performs the required casting
	    p_vnumber->operator[](current_index-1)=par_double;
	    break;
	  }
	case KeyArgument::ASCII :
	  {
	    vector<string>* p_vstring=(vector<string>*)current->p_object_variable;	// performs the required casting
	    p_vstring->operator[](current_index-1)=par_ascii;
	    break;
	  }
	  
	case KeyArgument::ASCIIlist :
	  {
	    IntVect* p_vnumber=(IntVect*)current->p_object_variable;	// performs the required casting
	    p_vnumber->operator[](current_index-1) =
	      find_in_ASCIIlist(par_ascii, *(current->p_object_list_of_values));
	    break;
	  }
	case KeyArgument::LIST_OF_INTS :
	  {
	    vector<IntVect>* p_matrixint=(vector<IntVect>*)current->p_object_variable;
	    p_matrixint->operator[](current_index-1)=par_intlist;
	    break;
	  }
	case KeyArgument::LIST_OF_DOUBLES :
	  {
	    vector<DoubleVect>* p_matrixdouble=(vector<DoubleVect>*)current->p_object_variable;
	    p_matrixdouble->operator[](current_index-1)=par_doublelist;
	    break;
	  }
	  /*	case LIST_OF_ASCII :
		{
		vector<string>* p_vectstring=(vector<string>*)current->p_object_variable;
		*p_vectstring=par_asciilist;
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
  // it was not in the list
  cerr << "Interfile warning : value of keyword \""
       << keyword << "\" is \""
       << par_ascii
       << "\"\n\tshould have been one of:";
  {
    for (unsigned int i=0; i<list_of_values.size(); i++)
      cerr << "\n\t" << list_of_values[i];
  }
  cerr << endl << endl;
  return -1;
}  	

int KeyParser::map_keyword(const string& keyword)
{
#if 0
  Keymap::iterator it;
  
  it=kmap.find(keyword);
  
  if(it!=kmap.end())
  {
    current=&(*it).second;
    return 1;
  }
  
  return 0;
#else
  current = find_in_keymap(keyword);
  return (current==0 ? 0 : 1);
#endif
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
        s << *reinterpret_cast<bool*>(i->second.p_object_variable) ? 1 : 0; break;
      case KeyArgument::ULONG:
        s << *reinterpret_cast<unsigned long*>(i->second.p_object_variable); break;
      case KeyArgument::NONE:
        break;
      case KeyArgument::ASCII :
	 s << *reinterpret_cast<string*>(i->second.p_object_variable); break;	  	  
      case KeyArgument::ASCIIlist :
        { // TODO check
	    const int index = *reinterpret_cast<int*>(i->second.p_object_variable);
            s << (*i->second.p_object_list_of_values)[index];	      
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

    s << ends;
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
    Line line;
    
    for (Keymap::const_iterator i=kmap.begin(); i!= kmap.end(); ++i)
    {
      
      if (i->second.p_object_member == &KeyParser::do_nothing ||
          i->second.p_object_member == &KeyParser::start_parsing ||
          i->second.p_object_member == &KeyParser::stop_parsing)
          continue;

      keyword = i->first;

      cout << keyword << " := ";
      {
        char buf[MAX_LINE_LENGTH];
        strcpy(buf, ":= ");
        cin.getline(buf+strlen(buf),MAX_LINE_LENGTH-strlen(buf),'\n');    
        line=buf;
      }

      parse_value_in_line(line, false);
      process_key();    
    }
    if (post_processing())    
       cout << " Asking all questions again! (Sorry)\n";
    else
      return;
  }
}
END_NAMESPACE_STIR
