//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class KeyParser

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/


#include "KeyParser.h"
#include "line.h"

#include <fstream>
#include <cstring>
#ifndef TOMO_NO_NAMESPACES
using std::ifstream;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_TOMO

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

map_element::~map_element()
{
}


map_element& map_element::operator=(const map_element& me)
{
  type=me.type;
  p_object_member=me.p_object_member;
  p_object_variable=me.p_object_variable;
  p_object_list_of_values=me.p_object_list_of_values;
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
  string::size_type cp =0;		//current index
  char const * const white_space = " \t_!";
  
  // skip white space
  cp=keyword.find_first_not_of(white_space,0);
  
  if(cp==string::npos)
    return string();
  
  // remove trailing white spaces
  const string::size_type eok=keyword.find_last_not_of(white_space);
  
  string kw;
  kw.reserve(eok+1);
  char previous_was_white_space = false;
  while (cp <= eok)
  {
    if (isspace(static_cast<int>(keyword[cp])) || keyword[cp]=='_' || keyword[cp]=='!')
    {
      if (!previous_was_white_space)
      {
        kw.append(1,' ');
        previous_was_white_space = true;
      }
      // else: skip this white space character
    }
    else
    {
      kw.append(1,static_cast<char>(tolower(static_cast<int>(keyword[cp]))));
      previous_was_white_space = false;
    }
    ++cp;
  }
  return kw;
}


void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			KeywordProcessor function,
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  kmap[standardise_keyword(keyword)] = 
    map_element(t, function, variable, list_of_values);
}

void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  kmap[standardise_keyword(keyword)] = 
    map_element(t, &KeyParser::set_variable, variable, list_of_values);
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
    
  if (parse_line(false))	
    process_key();
  if (status != parsing)
  { warning("KeyParser error: required first keyword not found\n");  return 1; }

  while(status==parsing)
    {
      if(parse_line(true))	
	process_key();    
    }
  
  return 0;
  
}	

int KeyParser::parse_line(const bool write_warning)
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
    // KT 10/7/2000 added
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
      line.get_param(par_ascii);
      break;
    case KeyArgument::INT :
      line.get_param(par_int);
      break;
    case KeyArgument::ULONG :
      line.get_param(par_ulong);
      break;
    case KeyArgument::DOUBLE :
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


void KeyParser::set_variable()
{
  // TODO this does not handle the vectored key convention
  
  // current_index is set to 0 when there was no index
  if(!current_index)
    {
      switch(current->type)
	{	  
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
	    IntVect* p_vectint=(IntVect*)current->p_object_variable;
	    *p_vectint=par_intlist;
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
  Keymap::iterator it;
  
  it=kmap.find(keyword);
  
  if(it!=kmap.end())
  {
    current=&(*it).second;
    return 1;
  }
  
  return 0;
  
}



void KeyParser::process_key()
{
  if(current->p_object_member!=NULL)
  {
    (this->*(current->p_object_member))();	//calls appropriate member function
  }
}

END_NAMESPACE_TOMO
