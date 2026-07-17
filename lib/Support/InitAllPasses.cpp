#include "ondrix/InitAllPasses.h"

#include "ondrix/Conversion/Passes.h"
#include "ondrix/Transforms/Passes.h"

void ondrix::registerAllOndrixPasses() {
  registerOndrixPasses();
  registerOndrixTransformsPasses();
}
