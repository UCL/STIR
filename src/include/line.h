
#ifndef __LINE_H__
#define __LINE_H__

// KT 14/12 changed <> to "" in next  include
#include "pet_string.h"
#include <vector.h>
#define LINE_ERROR -1
#define LINE_OK		0
#define MAX_LINE_LENGTH 256

class Line : private String
{

public :
	Line() : String()
	{}

	Line& operator=(const char* ch);

	String get_keyword();
	int get_index();
	int get_param(vector<int>& v);
	int get_param(vector<String>& v);
	int get_param(String& s);
	int get_param(int& i);
};

#endif

