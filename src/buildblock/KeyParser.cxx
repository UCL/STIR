
#include <fstream.h>
#include <KeyParser.h>
#include <line.h>
#include <pet_string.h>

#pragma warning(disable: 4786)

// map_element implementation;

map_element::map_element()
{
	type=NONE;
	p_object_member=NULL;
	p_object_variable=NULL;
}

map_element::~map_element()
{
}

map_element& map_element::operator()(argtype t, void (KeyParser::*pom)(),void* pov)
{
	type=t;
	p_object_member=pom;
	p_object_variable=pov;
	return *this;
}

map_element& map_element::operator=(const map_element& me)
{
	type=me.type;
	p_object_member=me.p_object_member;
	p_object_variable=me.p_object_variable;
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

	if(in_stream!=NULL)
	{
		in_stream->close();
		delete in_stream;
	}
	//	delete current;   
}

void KeyParser::Init()
{
	map_element m;

	kmap["!INTERFILE"]=					m(NONE,			NULL);
	kmap["!name of data file"]=			m(ASCII,		&KeyParser::OpenFileStream);
	kmap["image data byte order"]=		m(ASCII,		&KeyParser::SetVariable,&image_data_byte_order);
	kmap["!PET STUDY (General)"]=		m(NONE,			NULL);
	kmap["PET data type"]=				m(ASCIIlist,	&KeyParser::SetVariable,&pettype);
	kmap["data format"]=				m(ASCIIlist,	NULL);
	kmap["!number format"]=				m(ASCII,		&KeyParser::SetVariable,&number_format);
	kmap["!number of bytes per pixel"]=	m(NUMBER,		&KeyParser::SetVariable,&bytes_per_pixel);
	kmap["Number of dimensions"]=		m(NUMBER,		&KeyParser::ReadMatrixInfo,&number_of_dimensions);
	kmap["!matrix size"]=				m(NUMBERlist,	&KeyParser::SetVariable,&matrix_size);
	kmap["matrix axis label"]=			m(ASCII,		&KeyParser::SetVariable,&matrix_labels);
	kmap["number of time frames"]=		m(NUMBER,		&KeyParser::ReadFramesInfo,&number_of_time_frames);
	kmap["!PET STUDY (Emission data)"]=	m(NONE,			NULL);
	kmap["segment storage order"]=		m(NUMBERlist,	&KeyParser::SetVariable,&sqc);
	kmap["!IMAGE DATA DESCRIPTION"]=	m(NONE,			NULL);
	kmap["!index nesting level"]=		m(ASCIIlist,	&KeyParser::SetVariable,&index_nesting_level);
	kmap["!image scaling factor"]=		m(NUMBER,		&KeyParser::SetVariable,&image_scaling_factor);
	kmap["data offset in bytes"]=		m(LNUMBER,		&KeyParser::SetVariable,&data_offset);
	kmap["!END OF INTERFILE"]=			m(NONE,			&KeyParser::SetEndStatus);


}

int KeyParser::StartParsing()
{

	char buf[256];
	int i=0;


	Line line;
	
	// checks if INTERFILE HEADER :

	input->getline(buf,256,'\n');
	line=buf;
	keyword=line.get_keyword();	

	if(keyword!="!INTERFILE")
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
		
	}

	// prepares data for PETSinogramOfVolume Constructor

	while(matrix_labels[i]!="segment")
		i++;
	max_r_index=i;
	i=0;
	
	while(matrix_labels[i]!="z" && matrix_labels[i]!="angle")
		i++;
	if(matrix_labels[i]=="z")
		storage_order=BYSINO;
	else
		storage_order=BYVIEW;
	
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
		switch(current->type)							// depending on the par_type, gets the correct value from the line
		{											// and sets the right temporary variable
		case NONE :
			break;
		case ASCII :
			line.get_param(par_ascii);
			break;
		case NUMBER :
			line.get_param(par_number);
			break;
		case NUMBERlist :
			par_numberlist.clear();
			line.get_param(par_numberlist);
			break;
		case ASCIIlist :
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

void KeyParser::OpenFileStream()
{
	in_stream=new fstream(par_ascii.c_str(),ios::in);
	if (in_stream->fail() || in_stream->bad())
		printf("Error opening file\n"); 
	else
	{
		current->p_object_variable=in_stream;
	}
	
	printf("parameter : %s\n", par_ascii.c_str());
}

void KeyParser::ReadMatrixInfo()
{
	SetVariable();
	matrix_labels.resize(number_of_dimensions);
	matrix_size.resize(number_of_dimensions);
	
}

void KeyParser::ReadFramesInfo()
{
	SetVariable();
	image_scaling_factor.resize(number_of_time_frames);
	data_offset.resize(number_of_time_frames);
	
}

void KeyParser::SetVariable()
{
	if(!current_index)
	{
		switch(current->type)
		{
		case NUMBER :
			{
				int* p_number=(int*)current->p_object_variable;	// performs the required casting
				*p_number=par_number;
				break;
			}
		case ASCII :
			{
				String* p_string=(String*)current->p_object_variable;	// performs the required casting
				*p_string=par_ascii;
				break;
			}
		case NUMBERlist :
			{
				IntVect* p_vectint=(IntVect*)current->p_object_variable;
				*p_vectint=par_numberlist;
				break;
			}
		case ASCIIlist :
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
		case NUMBER :
			{
				IntVect* p_vnumber=(IntVect*)current->p_object_variable;	// performs the required casting
				p_vnumber->at(current_index-1)=par_number;
				break;
			}
		case ASCII :
			{
				vector<String>* p_vstring=(vector<String>*)current->p_object_variable;	// performs the required casting
				p_vstring->at(current_index-1)=par_ascii;
				break;
			}
		case NUMBERlist :
			{
				vector<IntVect>* p_matrixint=(vector<IntVect>*)current->p_object_variable;
				p_matrixint->at(current_index-1)=par_numberlist;
				break;
			}
	/*	case ASCIIlist :
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

	//if found
}



int KeyParser::ProcessKey()
{
	if(current->p_object_member!=NULL)
	{
		(this->*(current->p_object_member))();		//calls appropriate member function
		return 0;
	}
	return 1;										// no server function specified
}
