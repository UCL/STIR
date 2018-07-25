#ifndef ZOOMOPTIONS_H
#define ZOOMOPTIONS_H

#include "stir/common.h"

START_NAMESPACE_STIR

/*!
  \brief
  a class containing an enumeration type that can be used by functions to signal
  successful operation or not

  Example:
  \code
  Succeeded f() { do_something;  return Succeeded::yes; }
  void g() { if (f() == Succeeded::no) error("Error calling f"); }
  \endcode
*/
class ZoomOptions{
 public:
  enum ZO {preserve_sum, preserve_values, preserve_projections};
  ZoomOptions(const ZO& v) : v(v) {}
  private:
  ZO v;
};

END_NAMESPACE_STIR


#endif // ZOOMOPTIONS_H
