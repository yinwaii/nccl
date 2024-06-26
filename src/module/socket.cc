#include "nccl.h"
#include "core.h"
#include "socket.h"
#include "errors.h"
#include <cassert>

namespace ncclpp {
Socket::Socket(const SocketAddress* addr) {
	if (addr == nullptr) {
		throw Error("Socket address is null", ErrorCode::ncclInvalidArgument);
	} else {
		addr_ = addr->sa;
		/* IPv4/IPv6 support */
		int family = addr->sa.sa_family;
		fd_ = socket(family, SOCK_STREAM, 0);
		NCCLPPSYSCHECK(fd_, "socket");
	}
}

std::string Socket::toString() const {
	if (addr_.sa_family != AF_INET && addr_.sa_family != AF_INET6)
		return "";
	char host[NI_MAXHOST], service[NI_MAXSERV];
	(void)getnameinfo(&addr_, sizeof(SocketAddress), host, NI_MAXHOST, service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
	return std::string(host) + ":" + std::string(service);
}

void Socket::listen() {
	/* IPv4/IPv6 support */
	int salen = (addr_.sa_family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
	// localAddr port should be 0 (Any port)
  NCCLPPSYSCHECK(::bind(fd_, &addr_, salen), "bind");

  /* Get the assigned Port */
  socklen_t size = salen;
  NCCLPPSYSCHECK(getsockname(fd_, &addr_, &size), "getsockname");

  /* Put the socket in listen mode
   * NB: The backlog will be silently truncated to the value in /proc/sys/net/core/somaxconn
   */
  NCCLPPSYSCHECK(::listen(fd_, 16384), "listen");
}

void Socket::connect() {
	int ret;
  int timedout_retries = 0;
  int refused_retries = 0;
	int salen = (addr_.sa_family == AF_INET) ? sizeof(sockaddr_in) : sizeof(sockaddr_in6);
retry:
  SYSCHECKSYNC(::connect(fd_, &addr_, salen), "connect", ret);
	char line[1000];
	if (ret < 0) {
		if ((errno == ECONNREFUSED || errno == ETIMEDOUT)) {
			if ((errno == ECONNREFUSED && ++refused_retries < RETRY_REFUSED_TIMES) ||
					(errno == ETIMEDOUT && ++timedout_retries < RETRY_TIMEDOUT_TIMES)) {
				if (refused_retries % 1000 == 0) INFO(NCCL_ALL,"Call to connect %s to %s returned %s, retrying", toString().c_str(), socketToString(&addr_, line), strerror(errno));
				usleep(SLEEP_INT);
				goto retry;
			}
		}
		throw SysError("Connect to " + toString() + " failed ", errno);
	}
}

std::unique_ptr<Socket> Socket::accept() {
	SocketAddress sockaddr;
	socklen_t socklen = sizeof(sockaddr.sa);
	int fd = ::accept(fd_, &sockaddr.sa, &socklen);
	if (fd < 0) {
		throw SysError("Accept failed", errno);
	}
	return std::make_unique<Socket>(fd, &sockaddr);
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

void Socket::recv(void* ptr, int size) {
	int offset = 0;
	wait(NCCL_SOCKET_RECV, ptr, size, &offset);
}

void Socket::packedSend(void* ptr, int size) {
	send(&size, sizeof(size));
	send(ptr, size);
}

void Socket::packedRecv(void* ptr, int size) {
	int recvSize;
	recv(&recvSize, sizeof(size));
	if (recvSize != size) {
		throw Error("Message truncated : received " + std::to_string(recvSize) + " bytes instead of " + std::to_string(size), ErrorCode::ncclInternalError);
	}
	recv(ptr, std::min(recvSize, size));
}

void Socket::tmpPackedSend(socketAddress *addr, void* data, int size) {
  std::unique_ptr<Socket> tmpSendComm = Socket::connectAddress(addr);
	tmpSendComm->packedSend(data, size);
}

void Socket::tmpPackedRecv(Socket *sock, void* data, int size) {
	std::unique_ptr<Socket> tmpRecvComm = sock->accept();
	tmpRecvComm->packedRecv(data, size);
}

std::unique_ptr<Socket> Socket::createListenSocket(SocketAddress *localAddr) {
	std::unique_ptr<Socket> sock = std::make_unique<Socket>(localAddr);
	int sockfd = sock->getFd();

  if (sock->toPort()) {
    // Port is forced by env. Make sure we get the port.
    int opt = 1;
#if defined(SO_REUSEPORT)
    NCCLPPSYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)), "setsockopt");
#else
    NCCLPPSYSCHECK(setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)), "setsockopt");
#endif
  }
	sock->listen();
	localAddr->sa = sock->getAddr();

#ifdef ENABLE_TRACE
  TRACE(NCCL_INIT|NCCL_NET, "Listening on socket %s", sock->toString().c_str());
#endif

	return sock;
}

std::unique_ptr<Socket> Socket::connectAddress(SocketAddress* remoteAddr) {
	std::unique_ptr<Socket> sock = std::make_unique<Socket>(remoteAddr);
	int fd = sock->getFd();

	const int one = 1;
  NCCLPPSYSCHECK(setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char*)&one, sizeof(int)), "setsockopt");

  /*  const int bufsize = 128*1024;
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_SNDBUF, (char*)&bufsize, sizeof(int)), "setsockopt");
    SYSCHECK(setsockopt(fd, SOL_SOCKET, SO_RCVBUF, (char*)&bufsize, sizeof(int)), "setsockopt");*/

#ifdef ENABLE_TRACE
  // TRACE(NCCL_INIT|NCCL_NET, "Connecting to socket %s", sock->toString().c_str());
#endif

	sock->connect();
	// remoteAddr->sa = sock->getAddr();
	return sock;
}

}