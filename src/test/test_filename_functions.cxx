/*!

  \file
  \ingroup test

  \brief Test program for filename functions defined in utility.h

  \author Nikos Efthimiou
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000-2009, Hammersmith Imanet Ltd
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
#include "stir/FilePath.h"

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::string;
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
  { // The same with the new FilePath class. 
	  FilePath filename_with_directory("dir.name\\filename.v", false);

	  check(filename_with_directory.get_filename() == "filename.v");

	  filename_with_directory = "dir.name/filename.v"; 
	  check(filename_with_directory.get_filename() == "filename.v");

	  filename_with_directory = "a:filename.v";
	  check(filename_with_directory.get_filename() == "filename.v");

	  filename_with_directory = "filename.v";
	  check(filename_with_directory.get_filename() == "filename.v");
  }
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

    {
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
  }
   // Directory tests new tests
  {
      check(FilePath::is_absolute("\\bladi\\bla.v") == true);
      check(FilePath::is_absolute("bladi\\bla.v") == false);
      check(FilePath::is_absolute("/bladi/bla.v") == true);
      check(FilePath::is_absolute("a:/bladi/bla.v") == true);
      check(FilePath::is_absolute("bladi/bla.v") == false);
      check(FilePath::is_absolute("bla.v") == false);

      FilePath filename_with_directory("b\\c.v", false);

      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == "a\\b\\c.v");

      filename_with_directory = "b\\c.v";
      filename_with_directory.prepend_directory_name("a\\");
      check( filename_with_directory == "a\\b\\c.v");

      filename_with_directory = "b\\c.v";
      filename_with_directory.prepend_directory_name("a:");
      check( filename_with_directory == "a:b\\c.v");

      filename_with_directory = "c.v";
      filename_with_directory.prepend_directory_name("a\\b");
      check( filename_with_directory == "a\\b\\c.v");

      filename_with_directory = "\\b\\c.v";
      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == "\\b\\c.v");

      filename_with_directory = "b/c.v";
      filename_with_directory.prepend_directory_name("a/");
      check( filename_with_directory == "a/b/c.v");

      filename_with_directory = "b\\c.v";
      filename_with_directory.prepend_directory_name("a:");
      check( filename_with_directory == "a:b\\c.v");

      filename_with_directory = "c.v";
      filename_with_directory.prepend_directory_name("a/b");
      check( filename_with_directory == "a/b\\c.v");

      filename_with_directory = "/b/c.v";
      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == "/b/c.v");

  }

  //N.E: New directory tests.
  {
	  // No checks again because it will throw error.
	  FilePath fake_directory("dir.name\\filename", false);
	  check(FilePath::exist(fake_directory.get_path()) == false);

	  FilePath current_directory(FilePath::get_current_working_directory());
	  check(FilePath::exist(current_directory.get_path()) == true);
	  check(current_directory.is_directory() == true);
	  check(current_directory.is_writable() == true);

	  {
		  // Test create Path from Path.
		  // This is a bit of paradox so we have to set the first the
		  // checks to false.
		  // False, because not yet created.
		  FilePath path_to_append("my_test_folder_a", false);

		  FilePath newly_created_path = current_directory.append(path_to_append);

		  check(newly_created_path.is_directory() == true);
		  check(newly_created_path.is_writable() == true);

		  check(FilePath::exist(path_to_append.get_path()) == true);
	  }

	  {
		  // Test create Path from String.
		  string path_to_append("my_test_folder_b");

		  FilePath newly_created_path = current_directory.append(path_to_append);

		  check(newly_created_path.is_directory() == true);
		  check(newly_created_path.is_writable() == true);

		  // test for recrussive creation
		  string paths_to_append("my_test_folder_c\\my_test_folder_d");
		  FilePath newly_created_subfolder = newly_created_path.append(paths_to_append);

		  check(newly_created_subfolder.is_directory() == true);
		  check(newly_created_subfolder.is_writable() == true);
	  }
  }

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
  // N.E: same checks with Path class
{

 FilePath filename_with_directory("dir.name:filename", false);

 check( filename_with_directory.get_path()  == "dir.name:");

 filename_with_directory.add_extension(".img");
 check(filename_with_directory ==  "dir.name:filename.img");

 filename_with_directory =  "dir.name:filename.v";
 check(filename_with_directory == "dir.name:filename.v");

 filename_with_directory.add_extension(".img");

 // check no change made
 check(filename_with_directory ==  "dir.name:filename.v");

 // Replace is the proper action
 filename_with_directory.replace_extension(".img");
 check(filename_with_directory ==  "dir.name:filename.img");

 // N.E: Not sure about this. Set again in case of failure of the
 // previous test?
 filename_with_directory =  "dir.name:filename.v";
 check(filename_with_directory.get_filename() == "filename.v");

 filename_with_directory =  "filename.v";
 check(filename_with_directory.get_filename() == "filename.v");

}
  {
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
}
  // Directory tests new tests
  {
      check(FilePath::is_absolute(":bladi:bla.v") == true);
      check(FilePath::is_absolute("bladi:bla.v") == false);
      check(FilePath::is_absolute("bla.v") == false);

      FilePath filename_with_directory("b:c.v", false);

      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == "a:b:c.v");

      filename_with_directory = "b:c.v";
      filename_with_directory.prepend_directory_name("a:");
      check( filename_with_directory == "a:b:c.v");

      filename_with_directory = "c.v";
      filename_with_directory.prepend_directory_name("a:b");
      check( filename_with_directory == "a:b:c.v");

      filename_with_directory = ":b:c.v";
      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == ":b:c.v");
  }
  //N.E: New directory tests.
  {
      // No checks again because it will throw error.
      FilePath fake_directory("dir.name:filename", false);

     check (FilePath::exist(fake_directory.get_path()) == false);

      FilePath current_directory(FilePath::get_current_working_directory());
      check (FilePath::exist(current_directory.get_path()) == true);
      check(current_directory.is_directory() == true);
      check(current_directory.is_writable() == true);

      {
          // Test create Path from Path.
          // This is a bit of paradox so we have to set the first the
          // checks to false.
          // False, because not yet created.
          FilePath path_to_append("my_test_folder_a", false);

          FilePath newly_created_path = current_directory.append(path_to_append);

          check(newly_created_path.is_directory() == true);
          check(newly_created_path.is_writable() == true);

          check (FilePath::exist(path_to_append.get_path()) == true);
      }

      {
      // Test create Path from String.
      string path_to_append("my_test_folder_b");

      FilePath newly_created_path = current_directory.append(path_to_append);

      check(newly_created_path.is_directory() == true);
      check(newly_created_path.is_writable() == true);

      // test for recrussive creation
      string paths_to_append("my_test_folder_c:my_test_folder_d");
      FilePath newly_created_subfolder = newly_created_path.append(paths_to_append);

      check(newly_created_subfolder.is_directory() == true);
      check(newly_created_subfolder.is_writable() == true);
      }
  }
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
     // N.E: same checks with Path class
  {

    FilePath filename_with_directory("dir.name/filename", false);

    check( filename_with_directory.get_path()  == "dir.name/");

    filename_with_directory.add_extension(".img");
    check(filename_with_directory ==  "dir.name/filename.img");

    filename_with_directory =  "dir.name/filename.v";
    check(filename_with_directory == "dir.name/filename.v");

    filename_with_directory.add_extension(".img");

    // check no change made
    check(filename_with_directory ==  "dir.name/filename.v");

    // Replace is the proper action
    filename_with_directory.replace_extension(".img");
    check(filename_with_directory ==  "dir.name/filename.img");

    // N.E: Not sure about this. Set again in case of failure of the
    // previous test?
    filename_with_directory =  "dir.name/filename.v";
    check(filename_with_directory.get_filename() == "filename.v");

    filename_with_directory =  "filename.v";
    check(filename_with_directory.get_filename() == "filename.v");

  }
    // N.E: Finished the old tests with the new FilePath;

  // Directory tests old tests
  {
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
  }

  // Directory tests new tests
  {
      check(FilePath::is_absolute("/bladi/bla.v") == true);
      check(FilePath::is_absolute("bladi/bla.v") == false);
      check(FilePath::is_absolute("bla.v") == false);

      FilePath filename_with_directory("b/c.v", false);

      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == "a/b/c.v");

      filename_with_directory = "b/c.v";
      filename_with_directory.prepend_directory_name("a/");
      check( filename_with_directory == "a/b/c.v");

      filename_with_directory = "c.v";
      filename_with_directory.prepend_directory_name("a/b");
      check( filename_with_directory == "a/b/c.v");

      filename_with_directory = "/b/c.v";
      filename_with_directory.prepend_directory_name("a");
      check( filename_with_directory == "/b/c.v");
  }


  //N.E: New directory tests.
  {
      // No checks again because it will throw error.
      FilePath fake_directory("dir.name/filename", false);
      check (FilePath::exist(fake_directory.get_path()) == false);

      FilePath current_directory(FilePath::get_current_working_directory());
      check (FilePath::exist(current_directory.get_path()) == true);
      check(current_directory.is_directory() == true);
      check(current_directory.is_writable() == true);

      {
          // Test create Path from Path.
          // This is a bit of paradox so we have to set the first the
          // checks to false.
          // False, because not yet created.
          FilePath path_to_append("my_test_folder_a", false);

          FilePath newly_created_path = current_directory.append(path_to_append);

          check(newly_created_path.is_directory() == true);
          check(newly_created_path.is_writable() == true);

          check (FilePath::exist(path_to_append.get_path()) == true);
      }

      {
      // Test create Path from String.
      string path_to_append("my_test_folder_b");

      FilePath newly_created_path = current_directory.append(path_to_append);

      check(newly_created_path.is_directory() == true);
      check(newly_created_path.is_writable() == true);

      // test for recrussive creation
      string paths_to_append("my_test_folder_c/my_test_folder_d");
      FilePath newly_created_subfolder = newly_created_path.append(paths_to_append);

      check(newly_created_subfolder.is_directory() == true);
      check(newly_created_subfolder.is_writable() == true);
      }
  }

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
