//
// $Id$ :$Date$
//

#ifndef __KEYPARSER_H__
#define __KEYPARSER_H__

#include <map>
// KT 14/12 changed <String.h> to "pet_string.h"
// KT 09/08/98 forget about pet_string
#include <string>
#if !defined(__GNUG__) && !defined (__MSL__)
	using namespace std;
#endif

typedef string String;
//#include "pet_string.h"

#include <vector>
#include <fstream>
// KT 20/06/98 used for type_of_numbers and byte_order
#include "NumericInfo.h"


// KT 20/06/98 use this pragma only in VC++
#ifdef _MSC_VER
#pragma warning(disable: 4786)
#endif

#define		STATUS_END_PARSING	0
#define		STATUS_PARSING		1

#define		BYSINO				0
#define		BYVIEW				1

// KT 20/06/98 changed NUMBERlist to LIST_OF_INTS and (old) ASCIIlist to LIST_OF_ASCII
// (in Interfile 3.3 ASCIIlist really means a string which has to be in a specified list)
// KT 01/08/98 change number to int, lnumber to ulong, added double
typedef enum {NONE,ASCII,LIST_OF_ASCII,ASCIIlist, ULONG,INT,LIST_OF_INTS,DOUBLE,listASCIIlist} argtype;
typedef vector<int> IntVect;
typedef vector<unsigned long> UlongVect;
typedef vector<double> DoubleVect;


// KT 20/06/98 new
typedef vector<String> ASCIIlist_type;

class KeyParser;

class map_element
{
public :
	argtype type;
	void (KeyParser::*p_object_member)();	// pointer to a member function
	void* p_object_variable;		// pointer to a variable 
	const ASCIIlist_type *p_object_list_of_values;			// only used by ASCIIlist

	map_element();
	~map_element();
	
	// KT 20/06/98 added 4th argument for ASCIIlist
	map_element& operator()(argtype t, void (KeyParser::*pom)(),
	                        void* pov= NULL, const ASCIIlist_type *list = 0);
	map_element& operator=(const map_element& me);
};

// KT 20/06/98 gcc is up to date now, so we can use the 2 arg version of map<>
#if defined(__MSL__)
typedef map<String,map_element, less<String>, allocator<map_element> > Keymap;
#else
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

	// could be make into a union
	vector<String>  par_asciilist;
	// KT 01/08/98 change number to int, lnumber to ulong, added float
	IntVect		par_intlist;
	String		par_ascii;
	int			par_int;	
	double		par_double;	
	unsigned long par_ulong;

private :

	int ParseLine();
	int MapKeyword(String keyword);
	void ChangeStatus(int value);
	int ProcessKey();

private:

	// Lists of possible values for some keywords
	// KT 20/06/98 new
	ASCIIlist_type number_format_values;	
	// KT 01/08/98 new
	ASCIIlist_type byte_order_values;
	ASCIIlist_type type_of_data_values;
	ASCIIlist_type PET_data_type_values;	

	// Corresponding variables here

	int	number_format_index;
	int	byte_order_index;
	int type_of_data_index;
	int PET_data_type_index;

	// Extra private variables which will be translated to something more useful
	int	bytes_per_pixel;
	String	data_file_name;

public :

// TODO these shouldn't be here, but in PETStudy or something


	// 'Final' variables
	fstream*		in_stream;
	int			max_r_index;
	int			storage_order;
	// KT 20/06/98 new
	// This will be determined from number_format_index and bytes_per_pixel
	NumericType		type_of_numbers;
	// KT 01/08/98 new
	// This will be determined from byte_order_index, or just keep
	// its default value;
	ByteOrder file_byte_order;
	
	// KT 01/08/98 changed name to num_dimensions and num_time_frames
	int			num_dimensions;
	int			num_time_frames;
	vector<String>		matrix_labels;
	vector<IntVect>		matrix_size;
	DoubleVect		pixel_sizes;
	IntVect			sqc;
    // KT 01/08/98 changed to DoubleVect
	DoubleVect		image_scaling_factor;
	// KT 01/08/98 changed to UlongVect
	UlongVect		data_offset;
	
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
	// KT 20/06/98 new
	// This function is used by SetVariable only
	// It returns the index of a string in 'list_of_values', -1 of not found
	int find_in_ASCIIlist(const String& par_ascii, const ASCIIlist_type& list_of_values);
};

#endif
