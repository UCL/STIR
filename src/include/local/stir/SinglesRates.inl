#include "local/stir/SinglesRates.h"


START_NAMESPACE_STIR


const 
Scanner* SinglesRates::get_scanner_ptr() const
{ 
  return scanner_sptr.get();
}

END_NAMESPACE_STIR
