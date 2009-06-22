//
// $Id$
//
/*!

  \file
  \ingroup test

  \brief Test program for filename functions defined in utility.h

  \author Kris Thielemans
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
#include "stir/RunTests.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

/*!
  \brief Test class for filename functions defined in utility.h
  \ingroup test
*/
class FilenameTests : public RunTests
{
public:
  void run_tests();
};

void FilenameTests::run_tests()
{
  char filename_with_directory[max_filename_length];

  cerr << "Testing various filename utilities ";
#if defined(__OS_VAX__)

  cerr << "(using VAX-VMS filesystem conventions)" << endl;
  
  // relative names either contain no '[', or have '[.'
#if 0
  // tests disabled as add_extension is disabled
  strcpy(filename_with_directory, "[dir.name]filename");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "[dir.name]filename.img") == 0);
  strcpy(filename_with_directory, "[dir.name]filename.v");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "[dir.name]filename.v") == 0);
#endif

  strcpy(filename_with_directory, "[dir.name]filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
  {
    // same checks but with string versions
    string filename_with_directory = "[dir.name]filename";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory == "[dir.name]filename.img");
    filename_with_directory = "[dir.name]filename.v";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory == "[dir.name]filename.v");
    
    filename_with_directory = "[dir.name]filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
    filename_with_directory = "filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
  }
  check(is_absolute_pathname("da0:[bladi.b]bla.v") == true);
  check(is_absolute_pathname("[.bladi]bla.v") == false);
  check(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, "[.b]c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "da0:[a]"),
         "da0:[a.b]c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "da0:[a.b]"),
    "da0:[a.b]c.v") == 0);
  strcpy(filename_with_directory, "da0:[b]c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
    "da0:[b]c.v") == 0);

#elif defined(__OS_WIN__)

  cerr << "(using Windows filesystem conventions)" << endl;
  
  // relative names do not start with '\' or '?:\'
  // but we allow forward slashes as well
#if 0
  // tests disabled as add_extension is disabled
  strcpy(filename_with_directory, "dir.name\\filename");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name\\filename.img") == 0);
  strcpy(filename_with_directory, "dir.name\\filename.v");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name\\filename.v") == 0);
  strcpy(filename_with_directory, "dir.name/filename");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name/filename.img") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name/filename.v") == 0);
#endif
  strcpy(filename_with_directory, "dir.name\\filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "a:filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);

  {
    // same checks with string versions
    string filename_with_directory =  "dir.name\\filename";

    check(get_directory_name(filename_with_directory) == "dir.name\\");

    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name\\filename.img");
    filename_with_directory =  "dir.name\\filename.v";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name\\filename.v");
    filename_with_directory =  "dir.name/filename";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name/filename.img");
    filename_with_directory =  "dir.name/filename.v";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name/filename.v");

    filename_with_directory =  "dir.name\\filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
    filename_with_directory =  "dir.name/filename.v";

    check(get_directory_name(filename_with_directory) == "dir.name/");

    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
    filename_with_directory =  "a:filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
    filename_with_directory =  "filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
  }  
  check(is_absolute_pathname("\\bladi\\bla.v") == true);
  check(is_absolute_pathname("a:\\bladi\\bla.v") == true);
  check(is_absolute_pathname("bladi\\bla.v") == false);
  check(is_absolute_pathname("/bladi/bla.v") == true);
  check(is_absolute_pathname("a:/bladi/bla.v") == true);
  check(is_absolute_pathname("bladi/bla.v") == false);
  check(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, "b\\c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a\\b\\c.v") == 0);
  strcpy(filename_with_directory, "b\\c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a\\"),
         "a\\b\\c.v") == 0);
  strcpy(filename_with_directory, "b\\c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a:"),
         "a:b\\c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a\\b"),
         "a\\b\\c.v") == 0);
  strcpy(filename_with_directory, "\\b\\c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "\\b\\c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a\\b/c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a/"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a:"),
         "a:b/c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a/b"),
         "a/b\\c.v") == 0);
  strcpy(filename_with_directory, "/b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "/b/c.v") == 0);


#elif defined(__OS_MAC__)

  cerr << "(using MacOS filesystem conventions)" << endl;

  // relative names either have no ':' or do not start with ':'
#if 0
  // tests disabled as add_extension is disabled
  strcpy(filename_with_directory, "dir.name:filename");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name:filename.img") == 0);
  strcpy(filename_with_directory, "dir.name:filename.v");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name:filename.v") == 0);

  strcpy(filename_with_directory, "dir.name:filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
#endif

  {
    // same checks with string versions
    string filename_with_directory =  "dir.name:filename";

    check(get_directory_name(filename_with_directory) == "dir.name:");
    
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name:filename.img");
    filename_with_directory =  "dir.name:filename.v";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name:filename.v");

    filename_with_directory =  "dir.name:filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
    filename_with_directory =  "filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
  }  
  check(is_absolute_pathname("bladi:bla.v") == true);
  check(is_absolute_pathname(":bladi:bla.v") == false);
  check(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, ":b:c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, ":b:c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a:"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a:b:"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a:b"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, "b:c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "b:c.v") == 0);

#else // defined(__OS_UNIX__)

  cerr << "(using Unix filesystem conventions)" << endl;

#if 0
  // tests disabled as add_extension is disabled
  strcpy(filename_with_directory, "dir.name/filename");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name/filename.img") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "dir.name/filename.v") == 0);

  strcpy(filename_with_directory, "dir.name/filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
#endif  

  {
    // same checks with string versions
    string filename_with_directory =  "dir.name/filename";
    check(get_directory_name(filename_with_directory) == "dir.name/");

    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name/filename.img");
    filename_with_directory =  "dir.name/filename.v";
    add_extension(filename_with_directory, ".img");
    check(filename_with_directory ==  "dir.name/filename.v");

    filename_with_directory =  "dir.name/filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");
    filename_with_directory =  "filename.v";
    check(filename_with_directory.substr(
					 find_pos_of_filename(filename_with_directory),
					 string::npos)
	  ==  "filename.v");

  }

  check(is_absolute_pathname("/bladi/bla.v") == true);
  check(is_absolute_pathname("bladi/bla.v") == false);
  check(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, "b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a/"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a/b"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "/b/c.v");
  check(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "/b/c.v") == 0);
#endif /* Unix */  
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR
int main()
{
  FilenameTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
