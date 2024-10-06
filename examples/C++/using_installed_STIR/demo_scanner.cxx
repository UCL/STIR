#include "stir/Scanner.h"

int
main()
{
  std::cout << "Running demo_create_image" << std::endl;
  auto stir_scanner = stir::Scanner::get_scanner_from_name("D690");
  std::cout << "Constructed Scanner" << std::endl;
  // Basic test to ensure accessible scanner properties
  std::cout << stir_scanner->get_num_axial_blocks() << std::endl;
  std::cout << "Done" << std::endl;
  return EXIT_SUCCESS;
}