#include "stir/TextWriter.h"

START_NAMESPACE_STIR

aTextWriter* TextWriterHandle::information_channel_;
aTextWriter* TextWriterHandle::warning_channel_;
aTextWriter* TextWriterHandle::error_channel_;

void writeText(const char* text, OUTPUT_CHANNEL channel) {
	TextWriterHandle h;
    TextPrinter* pr = new TextPrinter();

	switch (channel) {
	case INFORMATION_CHANNEL:
        h.set_information_channel(pr);
		h.print_information(text);
		break;
	case WARNING_CHANNEL:
        h.set_warning_channel(pr);
		h.print_warning(text);
		break;
	case ERROR_CHANNEL:
        h.set_error_channel(pr);
		h.print_error(text);
	}
}

END_NAMESPACE_STIR
