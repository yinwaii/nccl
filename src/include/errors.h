// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef NCCLPP_ERRORS_H_
#define NCCLPP_ERRORS_H_

// #include "debug.h"
#include <sstream>
#include <stdexcept>
#include <string>

namespace ncclpp
{
	/// Enumeration of error codes used by MSCCL++.
	enum class ErrorCode
	{
		ncclSuccess = 0,
		ncclUnhandledCudaError = 1,
		ncclSystemError = 2,
		ncclInternalError = 3,
		ncclInvalidArgument = 4,
		ncclInvalidUsage = 5,
		ncclRemoteError = 6,
		ncclInProgress = 7,
		ncclNumResults = 8,
	};

	/// Convert an error code to a string.
	///
	/// @param error The error code to convert.
	/// @return The string representation of the error code.
	std::string errorToString(enum ErrorCode error);

	/// Base class for all errors thrown by MSCCL++.
	class BaseError : public std::runtime_error
	{
	public:
		/// Constructor for @ref BaseError.
		///
		/// @param message The error message.
		/// @param errorCode The error code.
		BaseError(const std::string &message, int errorCode);

		/// Constructor for @ref BaseError.
		///
		/// @param errorCode The error code.
		explicit BaseError(int errorCode);

		/// Virtual destructor for BaseError.
		virtual ~BaseError() = default;

		/// Get the error code.
		///
		/// @return The error code.
		int getErrorCode() const;

		/// Get the error message.
		///
		/// @return The error message.
		const char *what() const noexcept override;

	protected:
		std::string message_;
		int errorCode_;
	};

	/// A generic error.
	class Error : public BaseError
	{
	public:
		Error(const std::string &message, ErrorCode errorCode);
		virtual ~Error() = default;
		ErrorCode getErrorCode() const;
	};

	/// An error from a system call that sets `errno`.
	class SysError : public BaseError
	{
	public:
		SysError(const std::string &message, int errorCode);
		virtual ~SysError() = default;
	};

	// /// An error from a CUDA runtime library call.
	// class CudaError : public BaseError {
	//  public:
	//   CudaError(const std::string& message, int errorCode);
	//   virtual ~CudaError() = default;
	// };

	// /// An error from a CUDA driver library call.
	// class CuError : public BaseError {
	//  public:
	//   CuError(const std::string& message, int errorCode);
	//   virtual ~CuError() = default;
	// };

	// /// An error from an ibverbs library call.
	// class IbError : public BaseError {
	//  public:
	//   IbError(const std::string& message, int errorCode);
	//   virtual ~IbError() = default;
	// };

}; // namespace mscclpp

#define NCCLPPCHECK(call) do {  \
	ncclResult_t RES = call;                                              \
	if (RES != ncclSuccess && RES != ncclInProgress) {                    \
		throw ncclpp::Error("Error at " __FILE__":" __LINE__, static_cast<ncclpp::ErrorCode>(RES)); \
	}                                                                     \
} while (0);

// Check system calls
#define NCCLPPSYSCHECK(call, name) do { \
	int retval; \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
		throw ncclpp::SysError("Call to " name " failed : %s", errno); \
  } \
} while (false)

#define NCCLPPSYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)

#define CHECKBACKRET(call, ret) do { \
	try { ret = call; } \
	catch (const ncclpp::BaseError &e) { \
		WARN("%s", e.what()); \
		return static_cast<ncclResult_t>(e.getErrorCode()); \
	} \
} while (0);

#define CHECKBACK(call) do { \
	try { call; } \
	catch (const ncclpp::BaseError &e) { \
		WARN("%s", e.what()); \
		return static_cast<ncclResult_t>(e.getErrorCode()); \
	} \
} while (0);
#endif // NCCLPP_ERRORS_H_
