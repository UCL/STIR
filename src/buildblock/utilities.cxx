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
    Copyright (C) 2000- $Date$, IRSL
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

// KT 01/05/2000 moved here from .inl, after getting rid of CHARP template
bool ask (const string& str, bool default_value)
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


// KT 01/05/2000 new
string ask_string (const string& str, const string& default_value)
{   
  // TODO not nice to have a maximum length here
  char input[1000];
  
  cerr << "\n" << str 
       << "\n(Maximum string length is 1000)\n[default_value : \"" 
       << default_value
       << "\"]: \n";
  fgets(input,1000,stdin);
  if (strlen(input)==0)
    return default_value;
  else
  {
    // remove trailing newline
    if (strlen(input) < 1000)
	input[strlen(input)-1] = '\0';
    return input;
  }
}

const char * const 
find_filename(const char * const filename_with_directory)
{
  char *name;

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

// KT 14/01/2000 new
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

char *add_extension(char *file_in_directory_name, 
		    const char * const extension)

{
  if (strchr(find_filename(file_in_directory_name),'.') == NULL)
    strcat (file_in_directory_name,extension);
  return file_in_directory_name;
}

// SM&KT 18/01/2000 new
char *replace_extension(char *file_in_directory_name, 
		        const char * const extension)
{
  char * location_of_dot = 
    strchr(find_filename(file_in_directory_name),'.');

  // first truncate at extension
  if (location_of_dot!= NULL)
    *(location_of_dot) = '\0';

  strcat (file_in_directory_name,extension);
  return file_in_directory_name;
}

// KT 10/01/2000 new
bool
is_absolute_pathname(const char * const filename_with_directory)
{
#if defined(__OS_VAX__)
  // relative names either contain no '[', or have '[.'
  char * ptr = strchr(filename_with_directory,'[');
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
  char * ptr = strchr(filename_with_directory,':');
  if (ptr == NULL)
    return false;
  else
    return ptr != filename_with_directory;
#else // defined(__OS_UNIX__)
  // absolute names start with '/'
  return filename_with_directory[0] == '/';
#endif 
}


// KT 10/01/2000 new
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

char *ask_filename_with_extension(char *file_in_directory_name,
				  const char * const prompt,
				  const char * const default_extension)
{
  char ptr[max_filename_length];
  
  file_in_directory_name[0]='\0';
  while (strlen(file_in_directory_name)==0)
  { 
    printf ("\n%s ", prompt);
    if (strlen(default_extension))
      printf("(default extension '%s')", default_extension);
    printf(":");
    fgets(ptr,max_filename_length,stdin);
    sscanf(ptr," %s",file_in_directory_name);
  }
  add_extension(file_in_directory_name,default_extension);
  return(file_in_directory_name);
}


template <class FSTREAM>
void
ask_filename_and_open(FSTREAM& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed)
{
  char filename[max_filename_length];
  s.open(
    ask_filename_with_extension(filename, prompt, default_extension),
    mode);
  if (abort_if_failed && !s)
  { error("Error opening file %s\n", filename);  }
}

// instantiations

template 
void
ask_filename_and_open(ifstream& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);
template 
void
ask_filename_and_open(ofstream& s,
		      const char * const prompt,
	              const char * const default_extension,
		      ios::openmode mode,
		      bool abort_if_failed);
template 
void
ask_filename_and_open(fstream& s,
		      const char * const prompt,
	              const char * const default_extension,
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

void * read_stream_in_memory(istream& input, unsigned long& file_size)
{
  if (file_size == 0)
    file_size = find_remaining_size(input);
 
  cerr << "Reading " << file_size << " bytes from file." <<endl;

  // allocate memory
  char *memory = new char[file_size];
  if (memory == 0)
    { error("Not enough memory\n");  }

  {
    const unsigned long chunk_size = 1024*64;
    unsigned long to_read = file_size;
    char *current_location = memory;

    while( to_read != 0)
      {
	const unsigned long this_read_size = 
#ifndef STIR_NO_NAMESPACES
	  std::min(to_read, chunk_size);
#else
	  min(to_read, chunk_size);
#endif
	input.read(current_location, this_read_size);
	if (!input)
	{ error("Error after reading from stream\n");  }

	to_read -= this_read_size;
	current_location += this_read_size;
      }
  }
  return memory;
}

END_NAMESPACE_STIR
