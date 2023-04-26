/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
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
