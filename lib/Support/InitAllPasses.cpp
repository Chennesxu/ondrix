#include "ondrix/InitAllPasses.h"

#include "ondrix/Conversion/OndrixToOndsp/OndrixToOndsp.h"

void ondrix::registerAllOndrixPasses() { registerConversionPasses(); }
