//
// $Id$ :$Date$
//
/*!
  \file 
 
  \brief implementations for the KeyParser class

  \author Kris Thielemans
  \author Patrick Valente
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/

#include "Tomography_common.h"
#include "KeyParser.h"
#include "line.h"


#include <fstream>

START_NAMESPACE_TOMO

// map_element implementation;

map_element::map_element()
{
  type=KeyArgument::NONE;
  p_object_member=0;
  p_object_variable=0;
  p_object_list_of_values=0;
}

// KT 29/10/98 use typedef'ed name for 2nd arg
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
  // KT 28/07/98 new
  p_object_list_of_values=me.p_object_list_of_values;
  return *this;
}


// KeyParser implementation


// KT 13/11/98 moved istream arg to parse()
KeyParser::KeyParser()
{
  
  current_index=-1;
  status=end_parsing;
  current=new map_element();
}

KeyParser::~KeyParser()
{
}

// KT 13/11/98 new
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

// KT 13/11/98 moved istream arg from constructor
bool KeyParser::parse(istream& f)
{
  //KT 26/10/98 removed init_keys();
  input=&f;
  return (parse_header()==0 && post_processing()==0);
}


void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			KeywordProcessor function,
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  kmap[keyword.c_str()] = 
    map_element(t, function, variable, list_of_values);
}

void KeyParser::add_key(const string& keyword, 
			KeyArgument::type t, 
			void* variable,
			const ASCIIlist_type * const list_of_values)
{
  kmap[keyword.c_str()] = 
    map_element(t, &KeyParser::set_variable, variable, list_of_values);
}

int KeyParser::parse_header()
{
    
  if (parse_line(false))	
    process_key();
  // TODO skip white space ?
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
  
  // KT 22/10/98 changed EOF check
  if (!input->good())
  {
    // KT 22/10/98 added warning
    PETerror("KeyParser warning: early EOF or bad file");
    stop_parsing();
    return 0;
  }
  {
    char buf[MAX_LINE_LENGTH];
    input->getline(buf,MAX_LINE_LENGTH,'\n');
    line=buf;
  }
		// gets keyword
  keyword=line.get_keyword();
  current_index=line.get_index();
		// maps keyword to appropriate map_element (sets current)
  if(map_keyword(keyword))	
  {
    switch(current->type)	// depending on the par_type, gets the correct value from the line
    {				// and sets the right temporary variable
    case KeyArgument::NONE :
      break;
    case KeyArgument::ASCII :
      // KT 28/07/98 new
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

  // KT 22/10/98 warning message moved here
  // skip empty lines and comments
  if (keyword.length() != 0 && keyword[0] != ';' && write_warning)
    PETerror("KeyParser warning: unrecognized keyword: %s\n", keyword.c_str());

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
	  // KT 01/08/98 NUMBER->INT
	case KeyArgument::INT :
	  {
	    int* p_int=(int*)current->p_object_variable;	// performs the required casting
	    *p_int=par_int;
	    break;
	  }
	  // KT 01/08/98 new
	case KeyArgument::ULONG :
	  {
	    unsigned long* p_ulong=(unsigned long*)current->p_object_variable;	// performs the required casting
	    *p_ulong=par_ulong;
	    break;
	  }
	  // KT 01/08/98 new
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
	  // KT 20/06/98 new
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
	  // KT 29/10/98 added
	  PETerror("KeyParser error: unknown type. Implementation error");
	  break;
	}
    }
  else														// Sets vector elements using current_index
    {
      // KT 09/10/98 replaced vector::at with vector::operator[] 
      // KT 09/10/98 as SGI STL does not have at()
      switch(current->type)
	{
	  // KT 01/08/98 NUMBER->INT
	case KeyArgument::INT :
	  {
	    IntVect* p_vnumber=(IntVect*)current->p_object_variable;	// performs the required casting
	    p_vnumber->operator[](current_index-1)=par_int;
	    break;
	  }
	  // KT 01/08/98 new
	case KeyArgument::ULONG :
	  {
	    UlongVect* p_vnumber=(UlongVect*)current->p_object_variable;	// performs the required casting
	    p_vnumber->operator[](current_index-1)=par_ulong;
	    break;
	  }
	  // KT 01/08/98 new
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
	  // KT 20/06/98 new
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
	  // KT 29/10/98 added
	  PETerror("KeyParser error: unknown type. Implementation error");
	  break;
	}
    }
}

// KT 20/06/98 new
int KeyParser::find_in_ASCIIlist(const string& par_ascii, const ASCIIlist_type& list_of_values)
{
  {
    // TODO, once we know for sure type type of ASCIIlist_type, we could use STL find()
    for (int i=0; i<list_of_values.size(); i++)
      // TODO standardise to lowercase etc
      if (par_ascii == list_of_values[i])
	return i;
  }
  // it was not in the list
  cerr << "Interfile warning : value of keyword \""
       << keyword << "\" is \""
       << par_ascii
       << "\"\n\tshould have been one of:";
  {
    for (int i=0; i<list_of_values.size(); i++)
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
