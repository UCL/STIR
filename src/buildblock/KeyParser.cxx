//
// $Id$ :$Date$
//

// KT 01/08/98 removed ! from keywords
#include <iostream>
#include <fstream>
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
  type=NONE;
  p_object_member=NULL;
  p_object_variable=NULL;
  p_object_list_of_values=NULL;
}

map_element::~map_element()
{
}

map_element& map_element::operator()(argtype t, 
				     void (KeyParser::*pom)(),
				     void* pov,
				     const ASCIIlist_type *list_of_values)
{
  type=t;
  p_object_member=pom;
  p_object_variable=pov;
  p_object_list_of_values=list_of_values;
  return *this;
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

KeyParser::KeyParser()
{
  status=STATUS_PARSING;
  current_index=-1;
  in_stream=NULL;
  current=new map_element();
}

KeyParser::KeyParser(fstream* f)
{
  input=f;
  in_stream=NULL;
  current_index=-1;
  status=STATUS_PARSING;
  current=new map_element();
}

KeyParser::~KeyParser()
{
  // KT 01/08/98 leave this for calling program
  
#if 0
  if(in_stream!=NULL)
  {
    in_stream->close();
    delete in_stream;
  }
#endif
  // Note: 'current' is not allocated with 'new', so does not need a 'delete'
}

void KeyParser::Init()
{
  map_element m;
  
  // KT 20/06/98 new, unfortunate syntax...
  number_format_values.push_back("bit");
  number_format_values.push_back("ascii");
  number_format_values.push_back("signed integer");
  number_format_values.push_back("unsigned integer");
  number_format_values.push_back("float");
  
  // KT 01/08/98 new
  byte_order_values.push_back("LITTLEENDIAN");
  byte_order_values.push_back("BIGENDIAN");
  
  PET_data_type_values.push_back("Emission");
  PET_data_type_values.push_back("Transmission");
  PET_data_type_values.push_back("Blank");
  PET_data_type_values.push_back("AttenuationCorrection");
  PET_data_type_values.push_back("Normalisation");
  PET_data_type_values.push_back("Image");
  
  type_of_data_values.push_back("Static");
  type_of_data_values.push_back("Dynamic");
  type_of_data_values.push_back("Tomographic");
  type_of_data_values.push_back("Curve");
  type_of_data_values.push_back("ROI");
  type_of_data_values.push_back("PET");
  type_of_data_values.push_back("Other");
  
  // default values
  file_byte_order = ByteOrder::big_endian;
  type_of_data_index = 6; // PET
  
  kmap["INTERFILE"]=			m(NONE,		NULL);
  // KT 01/08/98 just set data_file_name variable now
  kmap["name of data file"]=		m(ASCII,	&KeyParser::SetVariable, &data_file_name);
  kmap["GENERAL DATA"]=			m(NONE,		NULL);
  kmap["GENERAL IMAGE DATA"]=		m(NONE,		NULL);
  kmap["type of data"]=			m(ASCIIlist,	&KeyParser::SetVariable,
    &type_of_data_index, 
    &type_of_data_values);
  
  kmap["image data byte order"]=	m(ASCIIlist,	&KeyParser::SetVariable,
    &byte_order_index, 
    &byte_order_values);
  kmap["PET STUDY (General)"]=		m(NONE,		NULL);
  kmap["PET data type"]=		m(ASCIIlist,	&KeyParser::SetVariable,
    &PET_data_type_index, 
    &PET_data_type_values);
  
  kmap["data format"]=			m(ASCII,	NULL);
  // KT 20/06/98 use ASCIIlist
  kmap["number format"]=		m(ASCIIlist,	&KeyParser::SetVariable,
    &number_format_index,
    &number_format_values);
  kmap["number of bytes per pixel"]=	m(INT,		&KeyParser::SetVariable,&bytes_per_pixel);
  kmap["number of dimensions"]=		m(INT,	&KeyParser::ReadMatrixInfo,&num_dimensions);
  kmap["matrix size"]=			m(LIST_OF_INTS,	&KeyParser::SetVariable,&matrix_size);
  kmap["matrix axis label"]=		m(ASCII,	&KeyParser::SetVariable,&matrix_labels);
  kmap["scaling factor (mm/pixel)"]=	m(DOUBLE,	&KeyParser::SetVariable,&pixel_sizes);
  kmap["number of time frames"]=	m(INT,	&KeyParser::ReadFramesInfo,&num_time_frames);
  kmap["PET STUDY (Emission data)"]=	m(NONE,		NULL);
  kmap["PET STUDY (Image data)"]=	m(NONE,		NULL);
  kmap["IMAGE DATA DESCRIPTION"]=	m(NONE,		NULL);
  kmap["image scaling factor"]=		m(DOUBLE,	&KeyParser::SetVariable,&image_scaling_factor);
  kmap["data offset in bytes"]=		m(ULONG,	&KeyParser::SetVariable,&data_offset);
  kmap["END OF INTERFILE"]=		m(NONE,		&KeyParser::SetEndStatus);
  
  
  
}



int KeyParser::StartParsing()
{
  
  char buf[256];
  int i=0;
  
  
  Line line;
  
  // checks if INTERFILE HEADER :
  
  // TODO replace with input >> line;
  input->getline(buf,256,'\n');
  line=buf;
  keyword=line.get_keyword();	
  
  if(keyword!="INTERFILE")
		{
    printf("This is not an Interfile");
    return 1;
		}
  else 
    printf("Ok :interfile! \n");
  
  // initializes keyword map
  Init();
  
  while(status)
  {
    if(ParseLine())	
      ProcessKey();
    // KT 01/08/98 added warning message
    else
      cerr << "Interfile Warning: unrecognized keyword: " << keyword << endl;
    
  }
  
  // KT 20/06/98 new
  type_of_numbers = NumericType(number_format_values[number_format_index], bytes_per_pixel);
  
  // KT 01/08/98 new
  file_byte_order = byte_order_index==0 ? 
    ByteOrder::little_endian : ByteOrder::big_endian;
  
  // This has to be here to be able to differentiate between
  // binary open or text open (for ASCII type).
  // TODO we don't support ASCII yet.
  in_stream = new fstream;
  open_read_binary(*in_stream, data_file_name.c_str());
  if(type_of_data_values[type_of_data_index] != "PET")
    cerr << "Interfile Warning: only 'type of data := PET' supported." << endl;
  
  if (PET_data_type_values[PET_data_type_index] != "Image")
  { cerr << "Interfile error: handling images only now." << endl;  return 1; }
  
  if (num_dimensions != 3)
  { cerr << "Interfile error: expecting 3D image " << endl; return 1; }
  
  if (matrix_labels[0]!="z" || matrix_labels[1]!="y" ||
    matrix_labels[2]!="x")
  { cerr << "Interfile: only supporting z,y,x order of coordinates now."
  << endl; return 1; }
  
  /*
  // prepares data for PETSinogramOfVolume Constructor
  
    
      // KT 28/07/98 stop if not there
      while(i<matrix_labels.size() && matrix_labels[i]!="segment")
      i++;
      if (i==matrix_labels.size())
      {
      cerr << "Matrix labels not in expected format.\n";
      return 0;
      }
      
	max_r_index=i;
	i=0;
	
	  while(matrix_labels[i]!="z" && matrix_labels[i]!="angle")
	  i++;
	  if(matrix_labels[i]=="z")
	  storage_order=BYSINO;
	  else
	  storage_order=BYVIEW;
  */
  
  return 0;
  
}	
void KeyParser::ChangeStatus(int value)
{
  status=value;
}

int KeyParser::ParseLine()
{
  char buf[MAX_LINE_LENGTH];
  Line line;
  int c=0;
  
  input->getline(buf,256,'\n');
  line=buf;
		// gets keyword
  keyword=line.get_keyword();
  current_index=line.get_index();
		// maps keyword to appropriate map_element (sets current)
  if(MapKeyword(keyword))	
  {
    switch(current->type)	// depending on the par_type, gets the correct value from the line
    {				// and sets the right temporary variable
    case NONE :
      break;
    case ASCII :
      // KT 28/07/98 new
    case ASCIIlist :
      line.get_param(par_ascii);
      break;
    case INT :
      line.get_param(par_int);
      break;
    case ULONG :
      line.get_param(par_ulong);
      break;
    case DOUBLE :
      line.get_param(par_double);
      break;
    case LIST_OF_INTS :
      par_intlist.clear();
      line.get_param(par_intlist);
      break;
    case LIST_OF_ASCII :
      par_asciilist.clear();
      line.get_param(par_asciilist);
      break;
    default :
      break;
    }
    return 1;
  }
  return 0;
}

void KeyParser::SetEndStatus()
{
  if(status==STATUS_PARSING)
    status=STATUS_END_PARSING;
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

void KeyParser::ReadMatrixInfo()
{
  SetVariable();
  matrix_labels.resize(num_dimensions);
  matrix_size.resize(num_dimensions);
  pixel_sizes.resize(num_dimensions);
  
}

void KeyParser::ReadFramesInfo()
{
  SetVariable();
  image_scaling_factor.resize(num_time_frames);
  data_offset.resize(num_time_frames);
  
}

void KeyParser::SetVariable()
{
  // TODO this does not handle the vectored key convention
  
  // current_index is set to 0 when there was no index
  if(!current_index)
  {
    switch(current->type)
    {
      // KT 01/08/98 NUMBER->INT
    case INT :
      {
	int* p_int=(int*)current->p_object_variable;	// performs the required casting
	*p_int=par_int;
	break;
      }
      // KT 01/08/98 new
    case ULONG :
      {
	unsigned long* p_ulong=(unsigned long*)current->p_object_variable;	// performs the required casting
	*p_ulong=par_ulong;
	break;
      }
      // KT 01/08/98 new
    case DOUBLE :
      {
	double* p_double=(double*)current->p_object_variable;	// performs the required casting
	*p_double=par_double;
	break;
      }
    case ASCII :
      {
	String* p_string=(String*)current->p_object_variable;	// performs the required casting
	*p_string=par_ascii;
	break;
      }
      // KT 20/06/98 new
    case ASCIIlist :
      {
	*((int *)current->p_object_variable) =
	  find_in_ASCIIlist(par_ascii, *(current->p_object_list_of_values));
	break;
      }
    case LIST_OF_INTS :
      {
	IntVect* p_vectint=(IntVect*)current->p_object_variable;
	*p_vectint=par_intlist;
	break;
      }
    case LIST_OF_ASCII :
      {
	vector<String>* p_vectstring=(vector<String>*)current->p_object_variable;
	*p_vectstring=par_asciilist;
	break;
      }
    default :
      break;
    }
  }
  else														// Sets vector elements using current_index
  {
    switch(current->type)
    {
      // KT 01/08/98 NUMBER->INT
    case INT :
      {
	IntVect* p_vnumber=(IntVect*)current->p_object_variable;	// performs the required casting
	p_vnumber->at(current_index-1)=par_int;
	break;
      }
      // KT 01/08/98 new
    case ULONG :
      {
	UlongVect* p_vnumber=(UlongVect*)current->p_object_variable;	// performs the required casting
	p_vnumber->at(current_index-1)=par_ulong;
	break;
      }
      // KT 01/08/98 new
    case DOUBLE :
      {
	DoubleVect* p_vnumber=(DoubleVect*)current->p_object_variable;	// performs the required casting
	p_vnumber->at(current_index-1)=par_double;
	break;
      }
    case ASCII :
      {
	vector<String>* p_vstring=(vector<String>*)current->p_object_variable;	// performs the required casting
	p_vstring->at(current_index-1)=par_ascii;
	break;
      }
      // KT 20/06/98 new
    case ASCIIlist :
      {
	IntVect* p_vnumber=(IntVect*)current->p_object_variable;	// performs the required casting
	p_vnumber->at(current_index-1) =
	  find_in_ASCIIlist(par_ascii, *(current->p_object_list_of_values));
	break;
      }
    case LIST_OF_INTS :
      {
	vector<IntVect>* p_matrixint=(vector<IntVect>*)current->p_object_variable;
	p_matrixint->at(current_index-1)=par_intlist;
	break;
      }
      /*	case LIST_OF_ASCII :
      {
      vector<String>* p_vectstring=(vector<String>*)current->p_object_variable;
      *p_vectstring=par_asciilist;
      break;
    }*/
    default :
      break;
    }
  }
}

// KT 20/06/98 new
int KeyParser::find_in_ASCIIlist(const String& par_ascii, const ASCIIlist_type& list_of_values)
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

int KeyParser::MapKeyword(String keyword)
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



int KeyParser::ProcessKey()
{
  if(current->p_object_member!=NULL)
  {
    (this->*(current->p_object_member))();		//calls appropriate member function
    return 0;
  }
  return 1;						// no server function specified
}
