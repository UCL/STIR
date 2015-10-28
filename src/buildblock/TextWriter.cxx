#include "TextWriter.h"

aTextWriter* TextWriterHandle::information_channel_;
aTextWriter* TextWriterHandle::warning_channel_;
aTextWriter* TextWriterHandle::error_channel_;

void writeText(const char* text, OUTPUT_CHANNEL channel) {
	TextWriterHandle h;
	switch (channel) {
	case INFORMATION_CHANNEL:
		h.print_information(text);
		break;
	case WARNING_CHANNEL:
		h.print_warning(text);
		break;
	case ERROR_CHANNEL:
		h.print_error(text);
	}
}

