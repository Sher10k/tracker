#ifndef HPP_VERSION_CONTROL
#define HPP_VERSION_CONTROL

#define XSTR(s) STR(s)

#define STR(s) #s

#ifndef VERSION
#define VERSION UNKNOWN
#endif

#include <string>

std::string version = XSTR(VERSION);

#endif
