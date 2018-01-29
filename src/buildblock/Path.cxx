#include "stir/Path.h"

START_NAMESPACE_STIR

Path::Path(){
    initSeparator();
}

Path::Path(const std::string &__str, bool _check)
{
    if(__str.size() == 0)
        error(boost::format("Path: Cannot initialise empty path."));

    my_string = __str;

    initSeparator();

    if (!_check)
        return;

    // Checks
    checks();
    // Check permissions
}

bool Path::isDirectory() const
{
    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        error(boost::format("Path: Cannot access %1%.")%my_string);
    else if( info.st_mode & S_IFDIR )
        return true;
    else
        return false;
}


bool Path::isRegFile() const
{
    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        error(boost::format("Path: Cannot access %1%.")%my_string);
    else if( info.st_mode & S_IFREG )
        return true;
    else
        return false;
}

bool Path::isWritable() const
{
    if( access(my_string.c_str(), 0) == 0 )
        return true;
    else
        return false;
}

bool Path::exist() const
{
    struct stat info;

    if( stat( my_string.c_str(), &info ) != 0 )
        return false;
    else
        return true;
}

Path Path::append(Path p)
{
    // Check permissions
    if (!isWritable())
       error(boost::format("Path: %1% is not writable.")%my_string);

    std::string new_path = my_string;
    //Check is directory
    if(!isDirectory())
    {
        if(!isRegFile())
        {
            error(boost::format("Path: Cannot find a directory in %1%.")%my_string);
        }
        else
        {
            new_path = getPath();
        }
    }

    std::vector<std::string> v{split(p.getPath(), separator.c_str())};

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

        const int dir_err = mkdir(new_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if (-1 == dir_err)
        {
            printf("Error creating directory!n");
            exit(1);
        }
    }
    return Path(new_path);
}

void
Path::addExtension(const std::string e)
{
    std::string::size_type pos = find_pos_of_extension();
    if (pos == std::string::npos)
        my_string += e;
}

std::string
Path::getPath() const
{
    std::size_t i = my_string.rfind(separator, my_string.length());
    if ((i) != std::string::npos) {
        return(my_string.substr(0, i+1));
    }

    return(my_string);
}

std::string
Path::getFileName() const
{
   std::size_t i = my_string.rfind(separator, my_string.length());
   if (i != std::string::npos) {
       return(my_string.substr(i+1, my_string.length() - i));
   }
   return(my_string);
}

std::string
Path::getFileExtension() const {
    std::size_t i = find_pos_of_extension();
    if (i != std::string::npos) {
        return(my_string.substr(i+1, my_string.length() - i));
    }
    return("");
}

void Path::checks() const
{
    struct stat s;
    if( stat(my_string.c_str(),&s) == 0 )
    {
        if ((s.st_mode & S_IFDIR ) &&
               ( s.st_mode & S_IFREG ) )
        {
            error(boost::format("Path: File %1% is neither a directory nor a file")%my_string);
        }
    }
    else
    {
        error(boost::format("Path: Maybe %1% does not exist?")%my_string);
    }
}

const std::vector<std::string> Path::split(const std::string& s, const char* c)
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
Path::find_pos_of_filename() const
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
Path::find_pos_of_extension() const
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

END_NAMESPACE_STIR
