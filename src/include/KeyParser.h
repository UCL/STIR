
#ifndef __KEYPARSER_H__
#define __KEYPARSER_H__

#include <map.h>
// KT 14/12 changed <String.h> to "pet_string.h"
#include "pet_string.h"

#include <vector.h>
#include <fstream.h>

#pragma warning(disable: 4786)

#define		STATUS_END_PARSING	0
#define		STATUS_PARSING		1

#define		BYSINO				0
#define		BYVIEW				1

typedef enum {NONE,ASCII,ASCIIlist,LNUMBER,NUMBER,NUMBERlist,listASCIIlist} argtype;
typedef vector<int> IntVect;

class KeyParser;

class map_element
{
public :
	argtype type;
	void (KeyParser::*p_object_member)();	// pointer to a member function
	void* p_object_variable;				// pointer to a variable 

	map_element();
	~map_element();
	
	map_element& operator()(argtype t, void (KeyParser::*pom)(),void* pov= NULL);
	map_element& operator=(const map_element& me);
};


#if defined (__GNUG__)
// KT 15/12 g++ doesn't have up-to-date STL yet...
typedef map<String,map_element, less<String> > Keymap;
#elif defined(__MSL__)
typedef map<String,map_element, less<String>, allocator<map_element> > Keymap;
#elif
typedef map<String,map_element> Keymap;
#endif

class KeyParser
{
private :					// varaibles
	Keymap kmap;
	fstream* input;
	int	status;				// parsing 0,end_parsing 1
	map_element* current;
	int current_index;
	String keyword;

	vector<String>  par_asciilist;
	IntVect			par_numberlist;
	String			par_ascii;
	int				par_number;	
	long			par_lnumber;

private :

	int ParseLine();
	int MapKeyword(String keyword);
	void ChangeStatus(int value);
	int ProcessKey();

public :

	// variables here
	fstream*			in_stream;
	int					max_r_index;
	int					storage_order;
	String				image_data_byte_order;
	String				number_format;
	int					bytes_per_pixel;
	int					number_of_dimensions;
	int					number_of_time_frames;
	vector<String>		matrix_labels;
	vector<IntVect>		matrix_size;
	IntVect				sqc;
//KT 14/12 changed from IntVect to vector<float>
	vector<float>		image_scaling_factor;
	vector<long int>	data_offset;
	vector<String>		pettype;
	vector<String>		index_nesting_level;
public : 

	// members

	KeyParser();
	KeyParser(fstream* f);
	~KeyParser();

	void Init();
	int StartParsing();
	
private :


	void ReadMatrixInfo();
	void ReadFramesInfo();
	void OpenFileStream();
	void SetEndStatus();
	void SetVariable();
};

#endif
