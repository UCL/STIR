//
// $Id$ :$Date$
//

// KT 01/08/98 removed ! from keywords
// KT 14/12 changed <> to "" in next 3 includes
#include "KeyParser.h"
#include "line.h"
// KT 20/06/98 needed for PETerror
#include "pet_common.h"

// KT 20/06/98 do this pragma only for VC++
#ifdef _MSC_VER
#pragma warning(disable: 4786)
#endif

// map_element implementation;

map_element::map_element()
{
  type=KeyArgument::NONE;
  p_object_member=0;
  p_object_variable=0;
  p_object_list_of_values=0;
}

map_element::map_element(KeyArgument::type t, 
			 void (KeyParser::*pom)(),
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

// KT 19/10/98 removed default constructor as unused at the moment
#if 0
KeyParser::KeyParser()
{
  // KT 15/10/98 added initialisation of input
  input=0;
  status=STATUS_PARSING;
  current_index=-1;
  in_stream=NULL;
  current=new map_element();
}
#endif // 0

// KT 16/10/98 changed to istream&
KeyParser::KeyParser(istream& f)
{
  input=&f;
  current_index=-1;
  status=end_parsing;
  current=new map_element();
}

KeyParser::~KeyParser()
{
}

bool KeyParser::parse()
{
  //KT 26/10/98 removed init_keys();
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
  if (status != parsing)
  { PETerror("KeyParser error: required first keyword not found\n");  return 1; }

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

#if 0
// KT 01/08/98 just set data_file_name variable now
void KeyParser::OpenFileStream()
{
  // TODO open as binary, except when type of data is ASCII...
  in_stream=new fstream(par_ascii.c_str(),ios::in);
  if (in_stream->fail() || in_stream->bad())
    printf("Error opening file\n"); 
  else
  {
    current->p_object_variable=in_stream;
  }
  
  printf("parameter : %s\n", par_ascii.c_str());
}
#endif


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
	case KeyArgument::LIST_OF_ASCII :
	  {
	    vector<string>* p_vectstring=(vector<string>*)current->p_object_variable;
	    *p_vectstring=par_asciilist;
	    break;
	  }
	default :
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
	  /*	case LIST_OF_ASCII :
		{
		vector<string>* p_vectstring=(vector<string>*)current->p_object_variable;
		*p_vectstring=par_asciilist;
		break;
		}*/
	default :
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
