//
// $Id$ :$Date$
//

#ifndef __LINE_H__
#define __LINE_H__

// KT 14/12 changed <> to "" in next  include
// KT 09/08/98 forget about pet_string
#include <string>
#if !defined(__GNUG__) && !defined (__MSL__)
	using namespace std;
#endif
typedef string String;

//#include "pet_string.h"
#include <vector>
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
	// KT 29/10/98 new
	int et_param(vector<double>& v);
	int get_param(vector<String>& v);
	int get_param(String& s);
	int get_param(int& i);
	// KT 01/08/98 new
	int get_param(unsigned long& i);
	// KT 01/08/98 new
	int get_param(double& i);
};

#endif

