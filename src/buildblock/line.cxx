//
// $Id$: $Date$
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class Line

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "line.h"


START_NAMESPACE_TOMO

const int LINE_ERROR =-1;
const int LINE_OK	= 0;

string Line::get_keyword()
{
	string kw;
	int sok,eok;		//start & end pos. of keyword
	int cp =0;		//current index
	
	// skip white space
	// KT 01/08/98 added \t for skippings tabs
	cp=find_first_not_of(" \t",0);


	if(cp!=string::npos)
	{
		// skip first !, as this is not strictly part of the keyword
		// KT 01/08/98 skip first ! now instead of space
		sok=find_first_not_of('!',cp);
		// keyword stops at either := or an index []
		cp=find_first_of(":[",sok);
		// remove trailing white spaces
		// KT 01/08/98 added \t for skippings tabs
		eok=find_last_not_of(" \t",--cp);
		// KT 01/08/98 replaced Mid by substr
		kw=substr(sok,eok-sok+1);
		//TODO standardise, e.g. make all the same case
	}

	return kw;
}
int Line::get_index()
{
	int cp,sok,eok;
	// we take 0 as a default value for the index
	int in=0;
	string sin;
	// KT 20/06/98 make sure that the index is part of the key (i.e. before :=)
	cp=find_first_of(":[",0);
	if(cp!=string::npos && operator[](cp) == '[')
	{
		sok=cp+1;
		eok=find_first_of(']',cp);
		// KT 09/08/98 check if closing bracket really there
		if (eok == string::npos)
		{
		  // TODO do something more graceful
		  warning("Interfile warning: invalid vectored key in line \n'%s'.\n%s",
		       this->c_str(), 
		       "Assuming this is not a vectored key.");
		  return 0;
		}
		// KT 09/08/98 replaced Mid by substr
		sin=substr(sok,eok-sok);
		in=atoi(sin.c_str());
	}
	return in;
}

int Line::get_param(string& s)
{
	int sok,eok;		//start & end pos. of keyword
	int cp =0;		//current index
	
	cp=find('=',0);

	if(cp!=string::npos)
	{
		cp++;
		sok=find_first_not_of(' ',cp);
		if(sok!=string::npos)
		{
			cp=length();
			// KT 09/08/98 added tab to list
			// strip trailing white space
			eok=find_last_not_of(" \t",cp);
			// KT 09/08/98 replace Mid by substr
			s=substr(sok,eok-sok+1);
			return LINE_OK;
		}
	}
	return LINE_ERROR;
}

int Line::get_param(int& i)
{
	string s;
	int r;

	r=get_param(s);
	if(r==LINE_OK)
		i=atoi(s.c_str());
	
	return r;
}

// KT 01/08/98 new
int Line::get_param(unsigned long& i)
{
	string s;
	int r;

	r=get_param(s);
	if(r==LINE_OK)
		// TODO not unsigned now
		i=atol(s.c_str());
	
	return r;
}

// KT 01/08/98 new
int Line::get_param(double& i)
{
	string s;
	int r;

	r=get_param(s);
	if(r==LINE_OK)
		i=atof(s.c_str());
	
	return r;
}



int Line::get_param(vector<int>& v)
{
	string s;
	// KT 02/11/98 don't use temporary variable anymore
	//int r=LINE_OK;
	int cp;
	int eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	// KT 20/06/98 has to be here now as I removed space from the list later on
	// skip white space
	cp=find_first_not_of(" \t",cp);
	// TODO? this does not check if brackets are balanced
	while (!end)
	{
		// KT 20/06/98 removed space from the list, as this allowed numbers separated by spaces
		cp=find_first_not_of("{},",cp);

		if(cp==string::npos)
		{
			end=true;
		}
		else
		{
			// KT 20/06/98 removed space from the list, as this allowed numbers separated by spaces
		        eop=find_first_of(",}",cp);
			if(eop==string::npos)
			{
				end=true;
				eop=length();
			}
			// KT 09/08/98 replaced Mid by substr
			s=substr(cp,eop-cp);
			// TODO use strstream, would allow templates
			// KT 02/11/98 don't use temporary variable anymore
			v.push_back(atoi(s.c_str()));
			cp=eop+1;
		}
	}
	return LINE_OK;
}

// KT 29/10/98 new
int Line::get_param(vector<double>& v)
{
	string s;
	// KT 02/11/98 don't use temporary variable anymore
	//int r=LINE_OK;
	int cp;
	int eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	// KT 20/06/98 has to be here now as I removed space from the list later on
	// skip white space
	cp=find_first_not_of(" \t",cp);
	// TODO? this does not check if brackets are balanced
	while (!end)
	{
		// KT 20/06/98 removed space from the list, as this allowed numbers separated by spaces
		cp=find_first_not_of("{},",cp);

		if(cp==string::npos)
		{
			end=true;
		}
		else
		{
			// KT 20/06/98 removed space from the list, as this allowed numbers separated by spaces
		        eop=find_first_of(",}",cp);
			if(eop==string::npos)
			{
				end=true;
				eop=length();
			}
			// KT 09/08/98 replaced Mid by substr
			s=substr(cp,eop-cp);
			// TODO use strstream, would allow templates
			// KT 02/11/98 don't use temporary variable anymore
			v.push_back(atof(s.c_str()));
			cp=eop+1;
		}
	}
	return LINE_OK;
}

int Line::get_param(vector<string>& v)
{
	string s;

	// KT 02/11/98 don't use temporary variable anymore
	//int r=LINE_OK;
	int cp;
	int eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	// KT 20/06/98 has to be here now as I removed space from the list later on
	// skip white space
	cp=find_first_not_of(" \t",cp);
	// TODO? this does not check if brackets are balanced
	while (!end)
	{
		// KT 20/06/98 removed space from the list
		cp=find_first_not_of("{},",cp);
		// KT 09/08/98 skip white space now
		cp=find_first_not_of(" \t",cp);

		if(cp==string::npos)
		{
			end=true;
		}
		else
		{
			// KT 20/06/98 removed space from the list
			eop=find_first_of(",}",cp);
			if(eop==string::npos)
			{
				end=true;
				eop=length();
			}
			// KT 20/06/98 this if() is never true anymore, as eop points to one of ",}"
			// I think it didn't handle multiple words between commas anyway
			/*
			if(operator[](eop)==' ')
			{
				char temp=find_first_not_of(' ',eop+1);
				if(operator[](temp)!=',' && operator[](temp)!='}')
					eop=find_first_of(",} ",temp);
			}
			*/
			// KT 09/08/98 trim ending white space
			size_type eop2 = find_last_not_of(" \t",eop);
			// KT 09/08/98 changed Mid to substr
			s=substr(cp,eop2-cp);

			v.push_back(s);
			cp=eop+1;
		}
	}
	return LINE_OK;
}

Line& Line::operator=(const char* ch)
{
	string::operator=(ch);
	return *this;
}

END_NAMESPACE_TOMO
