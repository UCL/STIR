//
// $Id$ :$Date$
//

#ifndef __LINE_H__
#define __LINE_H__

// KT 14/12 changed <> to "" in next  include
// KT 09/08/98 forget about pet_string
#include <string>
// KT 12/11/98 import pet_common.h for namespace stuff
#include "pet_common.h"

// KT 12/11/98 removed
// typedef string String;

//#include "pet_string.h"
#include <vector>
#define LINE_ERROR -1
#define LINE_OK		0
#define MAX_LINE_LENGTH 256

// KT 12/11/98 use string
class Line : private string
{

public :
	Line() : String()
	{}

	Line& operator=(const char* ch);

	String get_keyword();
	int get_index();
	int get_param(vector<int>& v);
	// KT 29/10/98 new
	int get_param(vector<double>& v);
	int get_param(vector<String>& v);
	int get_param(String& s);
	int get_param(int& i);
	// KT 01/08/98 new
	int get_param(unsigned long& i);
	// KT 01/08/98 new
	int get_param(double& i);
};

#endif

