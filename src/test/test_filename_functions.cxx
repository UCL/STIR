//
// $Id$: $Date$
//

#include "pet_common.h"
#include "utilities.h"

int main()
{
  char filename_with_directory[max_filename_length];

  cerr << "Testing various filename utilities ";
#if defined(__OS_VAX__)

  cerr << "(using VAX-VMS filesystem conventions)" << endl;
  
  // relative names either contain no '[', or have '[.'
  strcpy(filename_with_directory, "[dir.name]filename");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "[dir.name]filename.img") == 0);
  strcpy(filename_with_directory, "[dir.name]filename.v");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "[dir.name]filename.v") == 0);

  strcpy(filename_with_directory, "[dir.name]filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
  assert(is_absolute_pathname("da0:[bladi.b]bla.v") == true);
  assert(is_absolute_pathname("[.bladi]bla.v") == false);
  assert(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, "[.b]c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "da0:[a]"),
         "da0:[a.b]c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "da0:[a.b]"),
    "da0:[a.b]c.v") == 0);
  strcpy(filename_with_directory, "da0:[b]c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
    "da0:[b]c.v") == 0);

#elif defined(__OS_WIN__)

  cerr << "(using Windows filesystem conventions)" << endl;
  
  // relative names do not start with '\' or '?:\'
  // but we allow forward slashes as well

  strcpy(filename_with_directory, "dir.name\\filename");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name\\filename.img") == 0);
  strcpy(filename_with_directory, "dir.name\\filename.v");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name\\filename.v") == 0);
  strcpy(filename_with_directory, "dir.name/filename");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name/filename.img") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name/filename.v") == 0);

  strcpy(filename_with_directory, "dir.name\\filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "a:filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
  assert(is_absolute_pathname("\\bladi\\bla.v") == true);
  assert(is_absolute_pathname("a:\\bladi\\bla.v") == true);
  assert(is_absolute_pathname("bladi\\bla.v") == false);
  assert(is_absolute_pathname("/bladi/bla.v") == true);
  assert(is_absolute_pathname("a:/bladi/bla.v") == true);
  assert(is_absolute_pathname("bladi/bla.v") == false);
  assert(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, "b\\c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a\\b\\c.v") == 0);
  strcpy(filename_with_directory, "b\\c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a\\"),
         "a\\b\\c.v") == 0);
  strcpy(filename_with_directory, "b\\c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a:"),
         "a:b\\c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a\\b"),
         "a\\b\\c.v") == 0);
  strcpy(filename_with_directory, "\\b\\c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "\\b\\c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a\\b/c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a/"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a:"),
         "a:b/c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a/b"),
         "a/b\\c.v") == 0);
  strcpy(filename_with_directory, "/b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "/b/c.v") == 0);


#elif defined(__OS_MAC__)

  cerr << "(using MacOS filesystem conventions)" << endl;

  // relative names either have no ':' or do not start with ':'
  strcpy(filename_with_directory, "dir.name:filename");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name:filename.img") == 0);
  strcpy(filename_with_directory, "dir.name:filename.v");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name:filename.v") == 0);

  strcpy(filename_with_directory, "dir.name:filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
  assert(is_absolute_pathname("bladi:bla.v") == true);
  assert(is_absolute_pathname(":bladi:bla.v") == false);
  assert(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, ":b:c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, ":b:c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a:"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a:b:"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a:b"),
         "a:b:c.v") == 0);
  strcpy(filename_with_directory, "b:c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "b:c.v") == 0);

#else defined(__OS_UNIX__)

  cerr << "(using Unix filesystem conventions)" << endl;

  strcpy(filename_with_directory, "dir.name/filename");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name/filename.img") == 0);
  strcpy(filename_with_directory, "dir.name/filename.v");
  add_extension(filename_with_directory, ".img");
  assert(strcmp(filename_with_directory, "dir.name/filename.v") == 0);

  strcpy(filename_with_directory, "dir.name/filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  strcpy(filename_with_directory, "filename.v");
  assert(strcmp(find_filename(filename_with_directory), "filename.v") == 0);
  
  assert(is_absolute_pathname("/bladi/bla.v") == true);
  assert(is_absolute_pathname("bladi/bla.v") == false);
  assert(is_absolute_pathname("bla.v") == false);

  strcpy(filename_with_directory, "b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a/"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a/b"),
         "a/b/c.v") == 0);
  strcpy(filename_with_directory, "/b/c.v");
  assert(strcmp(prepend_directory_name(filename_with_directory, "a"),
         "/b/c.v") == 0);
#endif /* Unix */

  cerr << "End of tests. Everything ok !" << endl;
  return 0;
}
