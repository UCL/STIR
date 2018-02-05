#include "stir/FilePath.h"

START_NAMESPACE_STIR

FilePath::FilePath(){
    initSeparator();
}

FilePath::FilePath(const std::string &__str, bool _check)
{
    if(__str.size() == 0)
        error(boost::format("FilePath: Cannot initialise empty path."));

    my_string = __str;

    initSeparator();

    if (!_check)
        return;

    // Checks
    checks();
    // Check permissions
}

bool FilePath::is_directory() const
{
    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        error(boost::format("FilePath: Cannot access %1%.")%my_string);
    else if( info.st_mode & S_IFDIR )
        return true;
    else
        return false;
}


bool FilePath::is_regfile() const
{
    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        error(boost::format("FilePath: Cannot access %1%.")%my_string);
    else if( info.st_mode & S_IFREG )
        return true;
    else
        return false;
}

bool FilePath::is_writable() const
{
    if( access(my_string.c_str(), 0) == 0 )
        return true;
    else
        return false;
}

bool FilePath::exist(std::string s)
{
    struct stat info;

    if (s.size()>0)
    {
        if( stat( s.c_str(), &info ) != 0 )
            return false;
        else
            return true;
    }

    //    if( stat( my_string.c_str(), &info ) != 0 )
    //        return false;
    //    else
    //        return true;

}

FilePath FilePath::append(FilePath p)
{
    return append(p.get_path());
}

FilePath FilePath::append(std::string p)
{
    // Check permissions
    if (!is_writable())
       error(boost::format("FilePath: %1% is not writable.")%my_string);

    std::string new_path = my_string;
    //Check is directory
    if(!is_directory())
    {
        if(!is_regfile())
        {
            error(boost::format("FilePath: Cannot find a directory in %1%.")%my_string);
        }
        else
        {
            new_path = get_path();
        }
    }

    // try to accomondate multiple sub paths
    std::vector<std::string> v{split(p, separator.c_str())};

    for (int i = 0; i < v.size(); i++)
    {
        {
            char* end_of_root_name = &new_path.back();
            if (separator.compare(end_of_root_name))
                new_path.append(separator);
        }
        new_path.append(v.at(i));

        {
            char* end_of_root_name = &new_path.back();
            if (separator.compare(end_of_root_name))
                new_path.append(separator);
        }

        if (FilePath::exist(new_path) == true)
        {
            warning(boost::format("FilePath: Path %1% already exists.")%new_path);
            continue;
        }

        const int dir_err = mkdir(new_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }
    }
    return FilePath(new_path);
}

void
FilePath::add_extension(const std::string e)
{
    std::string::size_type pos = find_pos_of_extension();
    if (pos == std::string::npos)
        my_string += e;
}

void
FilePath::replace_extension(const std::string e)
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
    if ((i) != std::string::npos) {
        return(my_string.substr(0, i+1));
    }

    return(my_string);
}

std::string
FilePath::get_filename() const
{
   std::size_t i = my_string.rfind(separator, my_string.length());
   if (i != std::string::npos) {
       return(my_string.substr(i+1, my_string.length() - i));
   }
   return(my_string);
}

std::string
FilePath::get_extension() const {
    std::size_t i = find_pos_of_extension();
    if (i != std::string::npos) {
        return(my_string.substr(i+1, my_string.length() - i));
    }
    return("");
}

void FilePath::checks() const
{
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
}

const std::vector<std::string> FilePath::split(const std::string& s, const char* c)
{
    std::string buff{""};
     std::vector<std::string> v;

    for(auto n:s)
    {
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
        pos = my_string.find_last_of( ':');
#elif defined(__OS_WIN__)
    pos = my_string.find_last_of( '\\');
    if (pos==std::string::npos)
        pos = my_string.find_last_of( '/');
    if (pos==std::string::npos)
        pos = my_string.find_last_of( ':');
#elif defined(__OS_MAC__)
    pos = my_string.find_last_of( ':');
#else // defined(__OS_UNIX__)
    pos = my_string.find_last_of( '/');
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

bool
FilePath::is_absolute(const std::string _filename_with_directory)
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

void FilePath::prepend_directory_name(std::string p)
{

    if (FilePath::is_absolute(my_string) ||
            p.size() == 0)
        return;

//    char * new_name =
//            new char[strlen(filename_with_directory) + strlen(directory_name) + 4];
//    strcpy(new_name, directory_name);
//    char * end_of_new_name = new_name + strlen(directory_name)-1;
    std::string new_name;// = p + separator + my_string;

#if defined(__OS_VAX__)
    // relative names either contain no '[', or have '[.'
    if (my_string.first() != '[' ||
            p.back() != ']')
    else
    {
        // peel of the ][ pair
        p.pop_back();
    }
     new_name = p + separator + my_string;
#elif defined(__OS_WIN__)
    // append \ if necessary
//    if (p.back() != ':' && p.back() != '\\' &&
//            *p.back() != '/')
        new_name = merge(p, my_string);
#elif defined(__OS_MAC__)
    // relative names either have no ':' or do not start with ':'
    // append : if necessary
    if (p.back() != ':')
        p.push_back(":");
    // do not copy starting ':' of filename
    if (my_string.front() == ':')
        my_string.erase(0);
    new_name = p + separator + my_string;
#else // defined(__OS_UNIX__)
    // append / if necessary
        new_name = merge(p, my_string);
#endif
        my_string = new_name;
}

std::string FilePath::merge(std::string first, std::string sec)
{
    if (first.back() == *separator.c_str() && sec.front() == *separator.c_str())
    {
        first.pop_back();
        return first + sec;
    }
    else if ((first.back() == *separator.c_str() && sec.front() != *separator.c_str()) ||
             (first.back() != *separator.c_str() && sec.front() == *separator.c_str()))
    {
        return first + sec;
    }
    else /*( (first.back() != separator
               && sec.front() != separator))*/
    {
        return first + separator + sec;
    }
}

END_NAMESPACE_STIR
