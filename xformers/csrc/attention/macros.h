#pragma once

#ifdef _WIN32
#if defined(xformers_EXPORTS)
#define XFORMERS_API __declspec(dllexport)
#else
#define XFORMERS_API __declspec(dllimport)
#endif
#else
#define XFORMERS_API
#endif
