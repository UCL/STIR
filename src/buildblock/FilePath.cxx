/*
    Copyright (C) 2018, University of Hull

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

/*
 *  For the code transfered from the utilites.cxx
 *
 *
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2010, Hammersmith Imanet Ltd
    Copyright (C) 2014, University College London
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

#include "stir/FilePath.h"
#include "stir/utilities.h"

#include <boost/format.hpp>

#if defined(__OS_WIN__)
    #include <windows.h>
     #include<sys/types.h> // required for stat.h
    #include<direct.h>
#else
    #include <unistd.h>
#endif

#include <sys/stat.h>

START_NAMESPACE_STIR

FilePath::FilePath(){
    initSeparator();
}

FilePath::FilePath(const std::string &__str, bool _run_checks)
{
    if(__str.size() == 0)
        error(boost::format("FilePath: Cannot initialise empty path."));

    my_string = __str;

    initSeparator();

    run_checks = _run_checks;

    // Checks
    checks();
}

bool FilePath::is_directory() const
{
#if defined(__OS_WIN__)
	DWORD dwAttrib = GetFileAttributes(my_string.c_str());

	return 	(dwAttrib != INVALID_FILE_ATTRIBUTES && 
		dwAttrib & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        error(boost::format("FilePath: Cannot access %1%.")%my_string);
    else if( info.st_mode & S_IFDIR )
        return true;
#endif
    return false;
}


bool FilePath::is_regular_file() const
{

#if defined(__OS_WIN__)
	DWORD dwAttrib = GetFileAttributes(my_string.c_str());

	return (dwAttrib != INVALID_FILE_ATTRIBUTES && 
		(dwAttrib & FILE_ATTRIBUTE_NORMAL));
#else

    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        error(boost::format("FilePath: Cannot access %1%.")%my_string);
    else if( info.st_mode & S_IFREG )
        return true;
#endif
	return false;
}

bool FilePath::is_writable() const
{
#if defined(__OS_WIN__)
	DWORD dwAttrib = GetFileAttributes(my_string.c_str());

	return (dwAttrib != INVALID_FILE_ATTRIBUTES);
#else	
    if( access(my_string.c_str(), 0) == 0 )
        return true;
    else
        return false;
#endif
}

bool FilePath::exists(const std::string& s)
{
#if defined(__OS_WIN__)
	DWORD dwAttrib = GetFileAttributes(s.c_str());

	return (dwAttrib != INVALID_FILE_ATTRIBUTES);

#else
	struct stat info;

	if (s.size()>0)
	{
		if (stat(s.c_str(), &info) != 0)
			return false;
		else
			return true;
	}
	return false;
#endif
}

FilePath FilePath::append(const FilePath &p)
{
    return append(p.get_path());
}

FilePath FilePath::append(const std::string &p)
{
    // Check permissions
    if (!is_writable())
       error(boost::format("FilePath: %1% is not writable.")%my_string);

    std::string new_path = my_string;

    //Check if this a directory or it contains a filename, too
    if(!is_directory())
    {
        if(!is_regular_file())
        {
            error(boost::format("FilePath: Cannot find a directory in %1%.")%my_string);
        }
        else
        {
            new_path = get_path();
        }
    }

    // Try to accomondate multiple sub paths
    // Find if string p has more than one levels and store them in a vector.
    std::vector<std::string> v = split(p, separator.c_str());

    // Run over the vector creating the subfolders recrusively.
    for (unsigned int i = 0; i < v.size(); i++)
    {
        new_path = merge(new_path, v.at(i));
        FilePath::append_separator(new_path);

        // if current level already exists move to the next.
        if (FilePath::exists(new_path) == true)
        {
            warning(boost::format("FilePath: Path %1% already exists.")%new_path);
            continue;
        }
		int nError;
#if defined(__OS_WIN__)
		nError = _mkdir(new_path.c_str());
#else
		nError = mkdir(new_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
#endif

        if (-1 == nError)
        {
			error("FilePath: Error creating directory!");
        }
    }
    return FilePath(new_path);
}

void
FilePath::add_extension(const std::string &e)
{
    std::string::size_type pos = find_pos_of_extension();
    if (pos == std::string::npos)
        my_string += e;
}

void
FilePath::replace_extension(const std::string& e)
{
    std::string::size_type pos = find_pos_of_extension();
    if (pos != std::string::npos)
    {
        my_string.erase(pos);
        my_string += e;
    }
}


std::string
FilePath::get_path() const
{
    std::size_t i = my_string.rfind(separator, my_string.length());
    if (i != std::string::npos)
    {
        return(my_string.substr(0, i+1));
    }

    return(my_string);
}

std::string
FilePath::get_filename() const
{
	std::size_t i = 0; 
#if defined(__OS_WIN__)
	i = my_string.rfind(separator, my_string.length());
	if (i == std::string::npos)
		i = my_string.rfind('/', my_string.length());
	if (i == std::string::npos)
		i = my_string.rfind(':', my_string.length());
#else
    i = my_string.rfind(separator, my_string.length());
#endif

    if (i != std::string::npos)
    {
        return(my_string.substr(i+1, my_string.length() - i));
    }
    return(my_string);
}

std::string
FilePath::get_extension() const
{
    std::size_t i = find_pos_of_extension();
    if (i != std::string::npos)
    {
        return(my_string.substr(i+1, my_string.length() - i));
    }
    return("");
}

void FilePath::checks() const
{
    if (!run_checks)
        return;

#if defined(__OS_WIN__)
	DWORD dwAttrib = GetFileAttributes(my_string.c_str());

	if (dwAttrib != INVALID_FILE_ATTRIBUTES &&
		!(dwAttrib & FILE_ATTRIBUTE_DIRECTORY))
	{
		error(boost::format("FilePath: File %1% is neither a directory nor a file")%my_string);
	}

#else
    struct stat s;
    if( stat(my_string.c_str(),&s) == 0 )
    {
        if ((s.st_mode & S_IFDIR ) &&
               ( s.st_mode & S_IFREG ) )
        {
            error(boost::format("FilePath: File %1% is neither a directory nor a file")%my_string);
        }
    }
    else
    {
        error(boost::format("FilePath: Maybe %1% does not exist?")%my_string);
    }
#endif
}

//On Windows there's GetCurrentDirectory
std::string FilePath::get_current_working_directory()
{
    char buffer[max_filename_length];
    char *ptr = getcwd(buffer, sizeof(buffer));
    std::string s_cwd;
    if (ptr) 
        s_cwd = ptr;

    return s_cwd;
}

const std::vector<std::string> FilePath::split(const std::string& s, const char* c)
{
    std::string buff = "";
    std::vector<std::string> v;

    if (strlen(c) == 0)
        c = separator.c_str();

    for(unsigned int i = 0; i < s.size(); i++)
    {
        char n = s.at(i);

        if(n != *c) buff+=n; else
            if(n == *c && buff != "") { v.push_back(buff); buff = ""; }
    }
    if(buff != "") v.push_back(buff);

    return v;
}

std::string::size_type
FilePath::find_pos_of_filename() const
{
    std::string::size_type pos;

#if defined(__OS_VAX__)
    pos = my_string.find_last_of( ']');
    if (pos==std::string::npos)
        pos = my_string.find_last_of(separator);
#elif defined(__OS_WIN__)
    pos = my_string.find_last_of( '\\');
    if (pos==std::string::npos)
        pos = my_string.find_last_of( '/');
    if (pos==std::string::npos)
        pos = my_string.find_last_of( ':');
#else
    pos = my_string.find_last_of(separator);
#endif
    if (pos != std::string::npos)
        return pos+1;
    else
        return 0;
}

std::string::size_type
FilePath::find_pos_of_extension() const
{
  std::string::size_type pos_of_dot =
    my_string.find_last_of('.');
  std::string::size_type pos_of_filename =
    find_pos_of_filename();

  if (pos_of_dot >= pos_of_filename)
    return pos_of_dot;
  else
    return std::string::npos;
}

void FilePath::append_separator(std::string& s)
{
#if defined(__OS_VAX__)
        s += ":";
#elif defined(__OS_WIN__)
        s += "\\";
#elif defined(__OS_MAC__)
        s += ":" ;
#else // defined(__OS_UNIX__)
        s += "/" ;
#endif
}

bool
FilePath::is_absolute(const std::string& _filename_with_directory)
{
    const char* filename_with_directory = _filename_with_directory.c_str();
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

void FilePath::prepend_directory_name(const std::string &p)
{

    if (FilePath::is_absolute(my_string) ||
            p.size() == 0)
        return;

    std::string new_name;

#if defined(__OS_VAX__)
    // relative names either contain no '[', or have '[.'
    if (my_string[0] != '[' ||
            p[p.length()-1] != ']')
        else
    {
        // peel of the ][ pair
        p.erase[p.length()-1];
    }
    new_name = p + separator + my_string;
#elif defined(__OS_WIN__)
    new_name = merge(p, my_string);
#elif defined(__OS_MAC__)
    // relative names either have no ':' or do not start with ':'
    // append : if necessary
    if (p[p.length()-1] != ':')
        p.push_back(":");
    // do not copy starting ':' of filename
    if (my_string[0] == ':')
        my_string.erase(0);
    new_name = p + separator + my_string;
#else // defined(__OS_UNIX__)
    // append / if necessary
    new_name = merge(p, my_string);
#endif
    my_string = new_name;
}

std::string FilePath::merge(const std::string &first, const std::string &sec)
{
	std::string sep = separator; 

#if defined(__OS_WIN__)
	//Check for the appropriate separator.
	// Again, in utilies when windows all separators are checked. 
	if (first[first.length() - 1] != *sep.c_str())
	{
		if (first[first.length() - 1] == '/')
			sep = '/';
		if (first[first.length() - 1] == ':')
			sep = ':';
	}

	if (sec[0] != *sep.c_str())
	{
		if (sec[0] == '/')
			sep = '/';
		if (sec[0] == ':')
			sep = ':';
	}
#endif

    // Just append a separator
    if (sec.size() == 0)
        return first + sep;

    if (first[first.length()-1] == *sep.c_str() && sec[0] == *sep.c_str())
    {
        return first.substr(0, first.length()-1) + sec;
    }
    else if ((first[first.length()-1] == *sep.c_str() && sec[0] != *sep.c_str()) ||
             (first[first.length()-1] != *sep.c_str() && sec[0] == *sep.c_str()))
    {
        return first + sec;
    }
    else /*( (first.back() != separator
               && sec.front() != separator))*/
    {
        return first + sep + sec;
    }
}

END_NAMESPACE_STIR
