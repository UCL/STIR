#ifndef TEXT_WRITER_TYPES
#define TEXT_WRITER_TYPES

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
		else
			DEFAULT_STREAM << text;
	}
};

class TextWriterHandle {
public:
	void set_information_channel(aTextWriter* info) {
		init_();
		information_channel_ = info;
	}
	void set_warning_channel(aTextWriter* warn) {
		init_();
		warning_channel_ = warn;
	}
	void set_error_channel(aTextWriter* errr) {
		init_();
		error_channel_ = errr;
	}
	void print_information(const char* text) {
		init_();
		if (information_channel_)
			information_channel_->write(text);
	}
	void print_warning(const char* text) {
		init_();
		if (warning_channel_)
			warning_channel_->write(text);
	}
	void print_error(const char* text) {
		init_();
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