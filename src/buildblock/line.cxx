

// line.cxx
#include <line.h>
#include <stdio.h>
#include <vector.h>

String Line::get_keyword()
{
	String kw;
	int sok,eok;		//start & end pos. of keyword
	int cp =0;		//current index
	
	cp=find_first_not_of(' ',0);


	if(cp!=String::npos)
	{
		sok=find_first_not_of(' ',cp);
		cp=find_first_of(":[",sok);
		eok=find_last_not_of(' ',--cp);
		kw=Mid(sok,eok-sok+1);
	}

	return kw;
}
int Line::get_index()
{
	int cp,sok,eok;
	int in=0;
	String sin;
	cp=find_first_of('[',0);
	if(cp!=String::npos)
	{
		sok=cp+1;
		eok=find_first_of(']',cp);
		sin=Mid(sok,eok-sok);
		in=atoi(sin.c_str());
	}
	return in;
}

//int Line::get_param(vector<String>& v)
//{}

int Line::get_param(String& s)
{
	int sok,eok;		//start & end pos. of keyword
	int cp =0;		//current index
	
	cp=find('=',0);

	if(cp!=String::npos)
	{
		cp++;
		sok=find_first_not_of(' ',cp);
		if(sok!=String::npos)
		{
			cp=length();
			eok=find_last_not_of(' ',cp);
			s=Mid(sok,eok-sok+1);
			return LINE_OK;
		}
	}
	return LINE_ERROR;
}
int Line::get_param(int& i)
{
	String s;
	int r;

	r=get_param(s);
	if(r==LINE_OK)
		i=atoi(s.c_str());
	
	return r;
}

int Line::get_param(vector<int>& v)
{
	String s;
	int r=LINE_OK;
	int cp;
	int eop;
	bool end=false;

	cp=find_first_of('=',0)+1;
	while (!end)
	{
		cp=find_first_not_of("{}, ",cp);

		if(cp==String::npos)
		{
			end=true;
		}
		else
		{
			eop=find_first_of(",} ",cp);
			if(eop==String::npos)
			{
				end=true;
				eop=length();
			}
			s=Mid(cp,eop-cp);
			r=atoi(s.c_str());
			v.push_back(r);
			printf("%d\n",r);
			cp=eop+1;
		}
	}
	return r;
}

int Line::get_param(vector<String>& v)
{
	String s;
	int r=LINE_OK;
	int cp;
	int eop;
	bool end=false;
	char temp;

	cp=find_first_of('=',0)+1;
	while (!end)
	{
		cp=find_first_not_of("{}, ",cp);

		if(cp==String::npos)
		{
			end=true;
		}
		else
		{
			eop=find_first_of(",} ",cp);
			if(eop==String::npos)
			{
				end=true;
				eop=length();
			}
			if(operator[](eop)==' ')
			{
				temp=find_first_not_of(' ',eop+1);
				if(operator[](temp)!=',' && operator[](temp)!='}')
					eop=find_first_of(",} ",temp);
			}
			s=Mid(cp,eop-cp);
			v.push_back(s);
			printf("%s\n",s.c_str());
			cp=eop+1;
		}
	}
	return r;
}
Line& Line::operator=(const char* ch)
{
	String::operator=(ch);
	return *this;
}
