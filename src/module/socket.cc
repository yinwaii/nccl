#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "errors.h"

namespace ncclpp {
Socket Socket::createListenSocket(union socketAddress *localAddr) {
  /* IPv4/IPv6 support */
  int family = localAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Create socket and bind it to a port */
  int sockfd = socket(family, SOCK_STREAM, 0);
	NCCLPPSYSCHECK(sockfd, "socket");

  if (socketToPort(&localAddr->sa)) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    NCCLPPSYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#else
    NCCLPPSYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#endif
  }

  // localAddr port should be 0 (Any port)
  NCCLPPSYSCHECK(bind(sockfd, &localAddr->sa, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  NCCLPPSYSCHECK(getsockname(sockfd, &localAddr->sa, &size), "getsockname");

#ifdef ENABLE_TRACE
  char line[1024];
  TRACE(NCCL_INIT|NCCL_NET,"Listening on socket %s", socketToString(&localAddr->sa, line));
#endif

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  NCCLPPSYSCHECK(listen(sockfd, 16384), "listen");
	return Socket(sockfd);
}

Socket Socket::connectAddress(union socketAddress* remoteAddr) {
  /* IPv4/IPv6 support */
  int family = remoteAddr->sa.sa_family;
  int salen = (family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);

  /* Connect to a hostname / port */
  int fd = socket(family, SOCK_STREAM, 0);
	NCCLPPSYSCHECK(fd, "socket");

	const int one = 1;
  NCCLPPSYSCHECK(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

  char line[1024];
#ifdef ENABLE_TRACE
  TRACE(NCCL_INIT|NCCL_NET,"Connecting to socket %s", socketToString(&remoteAddr->sa, line));
#endif

  int ret;
  int timedout_retries = 0;
  int refused_retries = 0;
retry:
  SYSCHECKSYNC(connect(fd, &remoteAddr->sa, salen), "connect", ret);
	if (ret < 0) {
		if ((errno == ECONNREFUSED || errno == ETIMEDOUT)) {
			if ((errno == ECONNREFUSED && ++refused_retries < RETRY_REFUSED_TIMES) ||
					(errno == ETIMEDOUT && ++timedout_retries < RETRY_TIMEDOUT_TIMES)) {
				if (refused_retries % 1000 == 0) INFO(NCCL_ALL,"Call to connect returned %s, retrying", strerror(errno));
				usleep(SLEEP_INT);
				goto retry;
			}
		}
		std::string str = socketToString(&remoteAddr->sa, line);
		throw SysError("Connect to " + str + " failed ", errno);
	}
	return Socket(fd);
}

void Socket::progressOpt(int op, void* ptr, int size, int* offset, int block) {
	int bytes = 0;
	char* data = (char*)ptr;
	do {
		if (op == NCCL_SOCKET_RECV) bytes = ::recv(fd_, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
		if (op == NCCL_SOCKET_SEND) bytes = ::send(fd_, data+(*offset), size-(*offset), block ? 0 : MSG_DONTWAIT);
		if (op == NCCL_SOCKET_RECV && bytes == 0) {
			throw SysError("Net : Connection closed by remote peer", errno);
		}
		if (bytes == -1) {
			if (errno != EINTR && errno != EWOULDBLOCK && errno != EAGAIN) {
				throw SysError("Call to recv failed", errno);
			} else {
				bytes = 0;
			}
		}
		(*offset) += bytes;
	} while (bytes > 0 && (*offset) < size);
}

void Socket::progress(int op, void* ptr, int size, int* offset) {
	progressOpt(op, ptr, size, offset, 0);
}

void Socket::wait(int op, void* ptr, int size, int* offset) {
	while (*offset < size)
		progressOpt(op, ptr, size, offset, 1);
}

void Socket::send(void* ptr, int size) {
	int offset = 0;
	wait(NCCL_SOCKET_SEND, ptr, size, &offset);
}

void Socket::receive(void* ptr, int size) {
	int offset = 0;
	wait(NCCL_SOCKET_RECV, ptr, size, &offset);
}

}

ncclResult_t createListenSocket(int *fd, union socketAddress *localAddr) {
	ncclpp::Socket socket;
	CHECKBACKRET(ncclpp::Socket::createListenSocket(localAddr), socket);
	*fd = socket.getFd();
	return ncclSuccess;
}

ncclResult_t connectAddress(int* fd, union socketAddress* remoteAddr) {
	ncclpp::Socket socket;
	CHECKBACKRET(ncclpp::Socket::connectAddress(remoteAddr), socket);
	*fd = socket.getFd();
	return ncclSuccess;
}

ncclResult_t socketProgressOpt(int op, int fd, void* ptr, int size, int* offset, int block) {
	CHECKBACK(ncclpp::Socket(fd).progressOpt(op, ptr, size, offset, block));
	return ncclSuccess;
}

ncclResult_t socketProgress(int op, int fd, void* ptr, int size, int* offset) {
  CHECKBACK(ncclpp::Socket(fd).progress(op, ptr, size, offset));
	return ncclSuccess;
}

ncclResult_t socketWait(int op, int fd, void* ptr, int size, int* offset) {
	CHECKBACK(ncclpp::Socket(fd).wait(op, ptr, size, offset));
  return ncclSuccess;
}

ncclResult_t socketSend(int fd, void* ptr, int size) {
	CHECKBACK(ncclpp::Socket(fd).send(ptr, size));
  return ncclSuccess;
}

ncclResult_t socketReceive(int fd, void* ptr, int size) {
	CHECKBACK(ncclpp::Socket(fd).receive(ptr, size));
  return ncclSuccess;
}