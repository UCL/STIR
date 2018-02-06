//
//
/*!

  \file
  \ingroup buildblock

  \brief Implementations for class stir::Line

  \author Patrick Valente
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
#include "stir/line.h"


START_NAMESPACE_STIR

const int LINE_ERROR =-1;
const int LINE_OK	= 0;

string Line::get_keyword()
{
  // keyword stops at either := or an index []	
  // TODO should check that = follows : to allow keywords with colons in there
  const size_type eok =find_first_of(":[",0);
  return substr(0,eok);
}

int Line::get_index()
{
	size_type cp,sok,eok;
	// we take 0 as a default value for the index
	int in=0;
	string sin;
	// make sure that the index is part of the key (i.e. before :=)
	cp=find_first_of(":[",0);
	if(cp!=string::npos && operator[](cp) == '[')
	{
		sok=cp+1;
		eok=find_first_of(']',cp);
		// check if closing bracket really there
		if (eok == string::npos)
		{
		  // TODO do something more graceful
		  warning("Interfile warning: invalid vectored key in line \n'%s'.\n%s",
		       this->c_str(), 
		       "Assuming this is not a vectored key.");
		  return 0;
		}
		sin=substr(sok,eok-sok);
		in=atoi(sin.c_str());
	}
	return in;
}

int Line::get_param(string& s)
{
	size_type sok,eok;		//start & end pos. of keyword
	size_type cp =0;		//current index
	
	cp=find('=',0);

	if(cp!=string::npos)
	{
		cp++;
		sok=find_first_not_of(' ',cp);
		if(sok!=string::npos)
		{
			cp=length();
			// strip trailing white space
			eok=find_last_not_of(" \t",cp);
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
	size_type cp;
	size_type eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	// skip white space
	cp=find_first_not_of(" \t",cp);
	// TODO? this does not check if brackets are balanced
	while (!end)
	{
		cp=find_first_not_of("{},",cp);

		if(cp==string::npos)
		{
			end=true;
		}
		else
		{
		  eop=find_first_of(",}",cp);
			if(eop==string::npos)
			{
				end=true;
				eop=length();
			}
			s=substr(cp,eop-cp);
			// TODO use strstream, would allow templates
			v.push_back(atoi(s.c_str()));
			cp=eop+1;
		}
	}
	return LINE_OK;
}


int Line::get_param(vector<double>& v)
{
	string s;
	// KT 02/11/98 don't use temporary variable anymore
	//int r=LINE_OK;
	size_type cp;
	size_type eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	// skip white space
	cp=find_first_not_of(" \t",cp);
	// TODO? this does not check if brackets are balanced
	while (!end)
	{
		cp=find_first_not_of("{},",cp);

		if(cp==string::npos)
		{
			end=true;
		}
		else
		{
			eop=find_first_of(",}",cp);
			if(eop==string::npos)
			{
				end=true;
				eop=length();
			}
			s=substr(cp,eop-cp);
			// TODO use strstream, would allow templates
			v.push_back(atof(s.c_str()));
			cp=eop+1;
		}
	}
	return LINE_OK;
}

int Line::get_param(vector<string>& v)
{
	string s;

	size_type cp;
	size_type eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	// skip white space
	cp=find_first_not_of(" \t",cp);
	// TODO? this does not check if brackets are balanced
	while (!end)
	{
		cp=find_first_not_of("{},",cp);
		cp=find_first_not_of(" \t",cp);

		if(cp==string::npos)
		{
			end=true;
		}
		else
		{
			eop=find_first_of(",}",cp);
			if(eop==string::npos)
			{
				end=true;
				eop=length();
			}
			// trim ending white space
			size_type eop2 = find_last_not_of(" \t",eop);
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

END_NAMESPACE_STIR
