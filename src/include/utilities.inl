//
// $Id$: $Date$
//

// Implementations for utilities.h

template <class CHARP, class NUMBER>
inline NUMBER 
ask_num (CHARP str,
	 NUMBER minimum_value,
	 NUMBER maximum_value,
	 NUMBER default_value)
{ 
  
  while(1)
  { 
    char input[30];

    cerr << "\n" << str 
         << "[" << minimum_value << "," << maximum_value 
	 << " D:" << default_value << "]: ";
    fgets(input,30,stdin);
    istrstream ss(input);
    NUMBER value = default_value;
    ss >> value;
    if ((value>=minimum_value) && (maximum_value>=value))
      return value;
    cerr << "\nOut of bounds. Try again.";
  }
}

template <class CHARP>
inline bool ask (CHARP str, bool default_value)
{ 
  
  char input[30];
  
  cerr << "\n" << str 
       << " [Y/N D:" 
       << (default_value ? 'Y' : 'N') 
       << "]: ";
  fgets(input,30,stdin);
  if (strlen(input)==0)
    return default_value;
  char answer = input[0];
  if (default_value==true)
  {
    if (answer=='N' || answer == 'n')
      return false;
    else
      return true;
  }
  else
  {
    if (answer=='Y' || answer == 'y')
      return true;
    else
      return false;
    
  }
}



template <class IFSTREAM>
inline IFSTREAM& open_read_binary(IFSTREAM& s, 
				  const char * const name)
{
#if 0
  //KT 30/07/98 The next lines are only necessary (in VC 5.0) when importing 
  // <fstream.h>. We use <fstream> now, so they are disabled.

  // Visual C++ does not complain when opening a nonexisting file for reading,
  // unless using ios::nocreate
  s.open(name, ios::in | ios::binary | ios::nocreate); 
#else
  s.open(name, ios::in | ios::binary); 
#endif
  // KT 14/01/2000 added name of file in error message
  if (s.fail() || s.bad())
    { PETerror("Error opening file %s\n", name); Abort(); }
  return s;
}

template <class OFSTREAM>
inline OFSTREAM& open_write_binary(OFSTREAM& s, 
				  const char * const name)
{
    s.open(name, ios::out | ios::binary); 
    // KT 14/01/2000 added name of file in error message
    if (s.fail() || s.bad())
    { PETerror("Error opening file %s\n", name); Abort(); }
    return s;
}
