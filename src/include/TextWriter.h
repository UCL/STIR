#ifndef TEXT_WRITER_TYPES
#define TEXT_WRITER_TYPES

#include <string.h>

#include <fstream>
#include <iostream>
#include <string>

#define DEFAULT_STREAM std::cerr

enum OUTPUT_CHANNEL {INFORMATION_CHANNEL, WARNING_CHANNEL, ERROR_CHANNEL};

class aTextWriter {
public:
	virtual ~aTextWriter() {}
	virtual void write(const char* text) const = 0;
};

class TextPrinter : public aTextWriter {
public:
	TextPrinter(const char* s = 0) : _stream(0) {
		if (s) {
			if (strcmp(s, "stdout") == 0 || strcmp(s, "cout") == 0)
				_stream = 1;
			else if (strcmp(s, "stderr") == 0 || strcmp(s, "cerr") == 0)
				_stream = 2;
		}
	}
	virtual void write(const char* text) const {
		switch (_stream) {
		case 1:
			std::cout << text;
			break;
		case 2:
			std::cerr << text;
			break;
		default:
			DEFAULT_STREAM << text;
		}
	}
private:
	int _stream;
};

class TextWriter : public aTextWriter {
public:
	std::ostream* out;
	TextWriter(std::ostream* os = 0) : out(os) {}
	virtual void write(const char* text) const {
		if (out) {
			(*out) << text;
			(*out).flush();
		}
	}
};

class TextWriterHandle {
public:
	TextWriterHandle() {
		init_();
	}
	void set_information_channel(aTextWriter* info) {
		information_channel_ = info;
	}
	void* information_channel_ptr() {
		return (void*)information_channel_;
	}
	void set_warning_channel(aTextWriter* warn) {
		warning_channel_ = warn;
	}
	void* warning_channel_ptr() {
		return (void*)warning_channel_;
	}
	void set_error_channel(aTextWriter* errr) {
		error_channel_ = errr;
	}
	void* error_channel_ptr() {
		return (void*)error_channel_;
	}
	void print_information(const char* text) {
		if (information_channel_)
			information_channel_->write(text);
	}
	void print_warning(const char* text) {
		if (warning_channel_)
			warning_channel_->write(text);
	}
	void print_error(const char* text) {
		if (error_channel_)
			error_channel_->write(text);
	}

private:
	static aTextWriter* information_channel_;
	static aTextWriter* warning_channel_;
	static aTextWriter* error_channel_;
	static void init_() {
		static bool initialized = false;
		if (!initialized) {
			information_channel_ = 0;
			warning_channel_ = 0;
			error_channel_ = 0;
			initialized = true;
		}
	}
};

void writeText(const char* text, OUTPUT_CHANNEL channel = INFORMATION_CHANNEL);

#endif