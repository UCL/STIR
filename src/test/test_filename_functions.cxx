//
// $Id$: $Date$
//
/*!

  \file
  \ingroup test

  \brief Test program for filename functions defined in utility.h

  \author Kris Thielemans
  \author PARAPET project

  \date $Date$

  \version $Revision$
*/
#include "utilities.h"
#include "RunTests.h"

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_TOMO

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
  strcpy(filename_with_directory, "[dir.name]filename");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "[dir.name]filename.img") == 0);
  strcpy(filename_with_directory, "[dir.name]filename.v");
  add_extension(filename_with_directory, ".img");
  check(strcmp(filename_with_directory, "[dir.name]filename.v") == 0);

  strcpy(filename_with_directory, "[dir.name]filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
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

  strcpy(filename_with_directory, "dir.name\\filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "a:filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  check(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
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

#else defined(__OS_UNIX__)

  cerr << "(using Unix filesystem conventions)" << endl;

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

END_NAMESPACE_TOMO

USING_NAMESPACE_TOMO
int main()
{
  FilenameTests tests;
  tests.run_tests();
  return tests.main_return_value();
}
