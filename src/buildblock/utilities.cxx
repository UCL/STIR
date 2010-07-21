//
// $Id$
//
/*!
  \file 
 
  \brief non-inline implementations for utility.h

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
#include "stir/utilities.h"
#include "stir/IndexRange3D.h"
#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ifstream;
using std::ofstream;
using std::fstream;
using std::streampos;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

bool 
ask (const string& str, bool default_value)
{  
  string input;
  
  while (true)
    {
      cerr << "\n" << str 
	   << " [Y/N D:" 
	   << (default_value ? 'Y' : 'N') 
	   << "]: ";
      std::getline(std::cin, input);
      if (input.size()==0)
	return default_value;
      const char answer = input[0];
      switch (answer)
	{
	case 'N':
	case 'n':
	  return false;
	case 'Y':
	case 'y':
	  return true;
	default:
	  cerr << "\nPlease answer Y or N\n";
    }    
  }
}


string ask_string (const string& str, const string& default_value)
{   
  string input;
  
  cerr << "\n" << str 
       << "\n[default_value : \"" 
       << default_value
       << "\"]: \n";
  std::getline(std::cin, input);
  if (input.size()==0)
    return default_value;
  else
    return input;
}
FILE*& open_read_binary(FILE*& fptr, 
                        const string& name)
{
  fptr = fopen(name.c_str(), "rb"); 
  if (ferror(fptr))
    { error("Error opening file %s\n", name.c_str());  }
  return fptr;
}

FILE*& open_write_binary(FILE*& fptr, 
                        const string& name)
{
  fptr = fopen(name.c_str(), "wb"); 
  if (ferror(fptr))
    { error("Error opening file %s\n", name.c_str());  }
  return fptr;
}

void close_file(FILE*& fptr)
{
  fclose(fptr);
  fptr=0;
}

const char *
find_filename(const char * const filename_with_directory)
{
  const char * name;

#if defined(__OS_VAX__)
 name = strrchr(filename_with_directory,']');
 if (name==NULL)
   name = strrchr(filename_with_directory,':');
#elif defined(__OS_WIN__)
 name = strrchr(filename_with_directory,'\\');
 if (name==NULL)
   name = strrchr(filename_with_directory,'/');
 if (name==NULL)
   name = strrchr(filename_with_directory,':');
#elif defined(__OS_MAC__)
 name = strrchr(filename_with_directory,':');
#else // defined(__OS_UNIX__)
 name = strrchr(filename_with_directory,'/');
#endif 
 if (name!=NULL)
   // KT 10/01/2000 name++ changed to name+1
   return name+1;
 else
   return filename_with_directory;
}

string::size_type
find_pos_of_filename(const string& filename_with_directory)
{
  string::size_type pos;

#if defined(__OS_VAX__)
  pos = filename_with_directory.find_last_of( ']');
  if (pos==string::npos)
    pos = filename_with_directory.find_last_of( ':');
#elif defined(__OS_WIN__)
  pos = filename_with_directory.find_last_of( '\\');
  if (pos==string::npos)
    pos = filename_with_directory.find_last_of( '/');
  if (pos==string::npos)
    pos = filename_with_directory.find_last_of( ':');
#elif defined(__OS_MAC__)
  pos = filename_with_directory.find_last_of( ':');
#else // defined(__OS_UNIX__)
  pos = filename_with_directory.find_last_of( '/');
#endif 
  if (pos != string::npos)
    return pos+1;
  else
    return 0;
}


string
get_filename(const string& filename_with_directory)
{
  return 
    filename_with_directory.substr(find_pos_of_filename(filename_with_directory));
}

char *
get_directory_name(char *directory_name, 
		   const char * const filename_with_directory)
{
  size_t num_chars_in_directory_name =
    find_filename(filename_with_directory) - filename_with_directory;
  strncpy(directory_name, filename_with_directory, num_chars_in_directory_name);
  directory_name[num_chars_in_directory_name] = '\0';
  return directory_name;
}

string
get_directory_name(const string& filename_with_directory)
{
  return 
    filename_with_directory.substr(0, find_pos_of_filename(filename_with_directory));
}

string::size_type
find_pos_of_extension(const string& file_in_directory_name)
{
  string::size_type pos_of_dot =
    file_in_directory_name.find_last_of('.');
  string::size_type pos_of_filename =
    find_pos_of_filename(file_in_directory_name);
  if (pos_of_dot >= pos_of_filename)
    return pos_of_dot;
  else
    return string::npos;
}

#if 0
// terribly dangerous for memory overrun.
// will only work if enough memor was allocated
char *add_extension(char *file_in_directory_name, 
		    const char * const extension)

{
  if (strchr(find_filename(file_in_directory_name),'.') == NULL)
    strcat (file_in_directory_name,extension);
  return file_in_directory_name;
}
#endif

string& 
add_extension(string& file_in_directory_name, 
	      const string& extension)
{
  string::size_type pos =
    find_pos_of_extension(file_in_directory_name);
  if (pos == string::npos)
    file_in_directory_name += extension;
  return file_in_directory_name;
}



#if 0
// terribly dangerous for memory overrun.
// will only work if new extension is shorter than old
char *replace_extension(char *file_in_directory_name, 
		        const char * const extension)
{
  const char * location_of_dot = 
    strchr(find_filename(file_in_directory_name),'.');

  // first truncate at extension
  if (location_of_dot!= NULL)
    *(location_of_dot) = '\0';

  strcat (file_in_directory_name,extension);
  return file_in_directory_name;
}
#endif

string& 
replace_extension(string& file_in_directory_name, 
	      const string& extension)
{
  string::size_type pos =
    find_pos_of_extension(file_in_directory_name);
  if (pos != string::npos)
    file_in_directory_name.erase(pos);
  file_in_directory_name += extension;
      
  return file_in_directory_name;
}

bool
is_absolute_pathname(const char * const filename_with_directory)
{
#if defined(__OS_VAX__)
  // relative names either contain no '[', or have '[.'
  const char * const ptr = strchr(filename_with_directory,'[');
  if (ptr==NULL)
    return false;
  else
    return *(ptr+1) != '.';
#elif defined(__OS_WIN__)
  // relative names do not start with '\' or '?:\'
  if (filename_with_directory[0] == '\\' ||
      filename_with_directory[0] == '/')
    return true;
  else
    return (strlen(filename_with_directory)>3 &&
            filename_with_directory[1] == ':' &&
 	    (filename_with_directory[2] == '\\' ||
 	     filename_with_directory[2] == '/')
 	    );
#elif defined(__OS_MAC__)
  // relative names either have no ':' or do not start with ':'
  const char * const ptr = strchr(filename_with_directory,':');
  if (ptr == NULL)
    return false;
  else
    return ptr != filename_with_directory;
#else // defined(__OS_UNIX__)
  // absolute names start with '/'
  return filename_with_directory[0] == '/';
#endif 
}

bool
is_absolute_pathname(const string& filename_with_directory)
{
  return 
    is_absolute_pathname(filename_with_directory.c_str());
}

// Warning: this function assumes that filename_with_directory 
// points to sufficient allocated space to contain the new string
char *
prepend_directory_name(char * filename_with_directory, 
		       const char * const directory_name)
{
  if (is_absolute_pathname(filename_with_directory) ||
      directory_name == 0 ||
      strlen(directory_name) == 0)
    return filename_with_directory;

  char * new_name = 
    new char[strlen(filename_with_directory) + strlen(directory_name) + 4];
  strcpy(new_name, directory_name);
  char * end_of_new_name = new_name + strlen(directory_name)-1;


#if defined(__OS_VAX__)
  // relative names either contain no '[', or have '[.'
  if (filename_with_directory[0] != '[' || 
      *end_of_new_name != ']')
    strcat(new_name, filename_with_directory);
  else
  {
    // peel of the ][ pair
    *end_of_new_name = '\0';
    strcat(new_name, filename_with_directory+1);
  }
#elif defined(__OS_WIN__)
  // append \ if necessary
  if (*end_of_new_name != ':' && *end_of_new_name != '\\' &&
      *end_of_new_name != '/')
    strcat(new_name, "\\");
  strcat(new_name, filename_with_directory);
#elif defined(__OS_MAC__)
  // relative names either have no ':' or do not start with ':'
  // append : if necessary
  if (*end_of_new_name != ':')
    strcat(new_name, ":");
  // do not copy starting ':' of filename
  if (filename_with_directory[0] == ':')
    strcat(new_name, filename_with_directory+1);
  else
    strcat(new_name, filename_with_directory);
#else // defined(__OS_UNIX__)
  // append / if necessary
  if (*end_of_new_name != '/')
    strcat(new_name, "/");
  strcat(new_name, filename_with_directory);
#endif 

  strcpy(filename_with_directory, new_name);
  delete[] new_name;
  return filename_with_directory;
}

string
ask_filename_with_extension(
			    const string& prompt,
			    const string& default_extension)
{
  string file_in_directory_name;
  while (file_in_directory_name.size()==0)
  { 
    cerr << prompt;
    if (default_extension.size()!=0)
      {
	cerr << "(default extension '"
	     << default_extension
	     << "'):";
      }
    std::getline(std::cin, file_in_directory_name);
  }
  add_extension(file_in_directory_name,default_extension);
  return(file_in_directory_name);
}

char *
ask_filename_with_extension(char *file_in_directory_name,
			    const string& prompt,
			    const string& default_extension)
{
  const string answer =
    ask_filename_with_extension(prompt, default_extension);
  strcpy(file_in_directory_name, answer.c_str());
  return(file_in_directory_name);
}

template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const string& prompt,
	              const string& default_extension,
		      ios::openmode mode,
		      bool abort_if_failed)
{
  string filename =
        ask_filename_with_extension(prompt, default_extension);
  s.open(
	 filename.c_str(),
	 mode);
  if (abort_if_failed && !s)
  { error("Error opening file %s\n", filename.c_str());  }
}

// instantiations

template 
void
ask_filename_and_open(ifstream& s,
		      const string& prompt,
	              const string& default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);
template 
void
ask_filename_and_open(ofstream& s,
		      const string& prompt,
	              const string& default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);
template 
void
ask_filename_and_open(fstream& s,
		      const string& prompt,
	              const string& default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);

// find number of remaining characters
streamsize find_remaining_size (istream& input)
{
   streampos file_current_pos = input.tellg();
   input.seekg(0L, ios::end);
   streampos file_end = input.tellg();
   input.clear(); // necessary because seek past EOF ?
   input.seekg(file_current_pos);
   return file_end - file_current_pos;
}

void * read_stream_in_memory(istream& input, streamsize& file_size)
{
  if (file_size == 0)
    file_size = find_remaining_size(input);
 
  cerr << "Reading " << file_size << " bytes from file." <<endl;

  // allocate memory
  // TODO file_size could be longer than what size_t allows, but arrays cannot be longer
  char *memory = new char[static_cast<std::size_t>(file_size)];
  if (memory == 0)
    { error("Not enough memory\n");  }

  {
    const streamsize chunk_size = 1024*64;
    streamsize to_read = file_size;
    char *current_location = memory;

    while( to_read != 0)
      {
	const streamsize this_read_size = 
#ifndef STIR_NO_NAMESPACES
	  std::min(to_read, chunk_size);
#else
	  min(to_read, chunk_size);
#endif
	input.read(current_location, this_read_size);
	if (!input)
	{ error("Error after reading from stream");  }

	to_read -= this_read_size;
	current_location += this_read_size;
      }
  }
  return memory;
}

END_NAMESPACE_STIR
