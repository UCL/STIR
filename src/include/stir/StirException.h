#ifndef STIR_EXCEPTION
#define STIR_EXCEPTION

#include <string.h>

#include <exception>
#include <iostream>

class StirException : public std::exception {
public:
	StirException(const char* reason, const char* file, int line) {
		size_t len = strlen(reason) + 1;
		_reason = new char[len];
		memcpy(_reason, reason, len);
		len = strlen(file) + 1;
		_file = new char[len];
		memcpy(_file, file, len);
		_line = line;
	}
	virtual ~StirException() {
		delete[] _reason;
		delete[] _file;
	}
	virtual const char* what() const throw()
	{
		return _reason;
	}
	const char* file() const throw()
	{
		return _file;
	}
	int line() const throw() {
		return _line;
	}
private:
	char* _reason;
	char* _file;
	int _line;
};

#endif