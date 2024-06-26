#include "bootstrap.h"
#include "core.h"
#include "errors.h"
#include "nccl.h"
#include "net.h"
#include "socket.h"
#include "utils.h"
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <list>

namespace ncclpp
{
static void setFilesLimit() {
  struct rlimit filesLimit;
  NCCLPPSYSCHECK(getrlimit(RLIMIT_NOFILE, &filesLimit), "getrlimit");
  filesLimit.rlim_cur = filesLimit.rlim_max;
  NCCLPPSYSCHECK(setrlimit(RLIMIT_NOFILE, &filesLimit), "setrlimit");
}

struct ExtInfo {
  int rank;
  int nranks;
  SocketAddress extHandleListenRoot;
  SocketAddress extHandleListen;
};

struct InternalUniqueId {
  int magic;
  SocketAddress addr;
};

enum bootstrapInterface_t { findSubnetIf = -1, dontCareIf = -2 };
class TcpBootstrap::Impl {
 public:
   static InternalUniqueId createUniqueId();
   InternalUniqueId getUniqueId() { return uniqueId_; }
   Impl(int rank, int nRanks);
   ~Impl();

   void initialize(InternalUniqueId *uniqueId);
   int getRank() { return rank_; }
   int getNranks() { return nRanks_; }
   void send(void *data, int size, int peer);
   void recv(void *data, int size, int peer);
   void allGather(void *allData, int size);
   void close();

 private:
   int rank_;
   int nRanks_;
   InternalUniqueId uniqueId_;
   std::thread rootThread_;
   SocketAddress addr_;
   std::unique_ptr<Socket> listenSock_;
   std::unique_ptr<Socket> ringSendSock_;
   std::unique_ptr<Socket> ringRecvSock_;
   std::vector<SocketAddress> peersAddr_;
   std::list<std::pair<int, std::unique_ptr<Socket>>> sockQueue_;
   // UniqueId uniqueId_;

   static SocketAddress getAddress(InternalUniqueId *uniqueId = nullptr);
   void root();
   void createRoot();
};

SocketAddress TcpBootstrap::Impl::getAddress(InternalUniqueId *uniqueId) {
  SocketAddress resAddr;
  char ifName[MAX_IF_NAME_SIZE];
  char *env = getenv("NCCL_COMM_ID");
  if (env) {
    if (uniqueId) {
      // handle stores a remote address
      // need to find a local addr that is in the same network as the remote addr
      if (findInterfaceMatchSubnet(ifName, &resAddr, &uniqueId->addr, MAX_IF_NAME_SIZE, 1) <= 0) {
        throw Error("NET/Socket : No usable listening interface found", ErrorCode::ncclSystemError);
      }
    }
    else {
      INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
      if (GetSocketAddrFromString(&resAddr, env) != 0) {
        throw Error("Invalid NCCL_COMM_ID, please use format: <ipv4>:<port> or [<ipv6>]:<port> or <hostname>:<port>", ErrorCode::ncclInvalidArgument);
      }
    }
  }
  else {
    if (findInterfaces(ifName, &resAddr, MAX_IF_NAME_SIZE, 1) <= 0) {
      throw Error("NET/Socket : No usable listening interface found", ErrorCode::ncclSystemError);
    }
  }
  return resAddr;
}

InternalUniqueId TcpBootstrap::Impl::createUniqueId() {
  InternalUniqueId uniqueId;
  // uniqueId.magic = NCCL_UNIQUE_ID_MAGIC;
  uniqueId.addr = getAddress();
  std::unique_ptr<Socket> sock = Socket::createListenSocket(&uniqueId.addr);
  return uniqueId;
}

TcpBootstrap::Impl::~Impl() {
  if (rootThread_.joinable()) {
    rootThread_.join();
  }
}

TcpBootstrap::Impl::Impl(int rank, int nRanks) : rank_(rank), nRanks_(nRanks) {
  TRACE(NCCL_INIT, "TcpBootstrap::Impl rank %d nranks %d", rank, nRanks);
  peersAddr_.resize(nRanks_);
  if (rank == 0)
    createRoot();
}

void TcpBootstrap::Impl::root() {
  // socketAddress rootAddr = getAddress();
  std::unique_ptr<Socket> rootSock = Socket::createListenSocket(&uniqueId_.addr);
  std::vector<SocketAddress> rankHandles(nRanks_);
  std::vector<SocketAddress> rankHandlesRoot(nRanks_); // for initial rank <-> root information exchange
  setFilesLimit();

  TRACE(NCCL_INIT, "Root@@BEGIN");
  /* Receive addresses from all ranks */
  int c = 0;
  do {
    SocketAddress zero = {0}; // for sanity checking
    ExtInfo info;
    TRACE(NCCL_INIT, "Root recving @@ %d, root addr %s", rank_, rootSock->toString().c_str());
    Socket::tmpPackedRecv(rootSock.get(), &info, sizeof(info));
    TRACE(NCCL_INIT, "Root recv msg from @@ %d", info.rank);

    if (nRanks_ != info.nranks) {
      throw Error("Bootstrap Root : mismatch in rank count from procs " + std::to_string(nRanks_) + " : " + std::to_string(info.nranks), ErrorCode::ncclInternalError);
    }

    if (memcmp(&zero, &rankHandlesRoot[info.rank], sizeof(SocketAddress)) != 0) {
      throw Error("Bootstrap Root : rank " + std::to_string(info.rank) + " of " + std::to_string(nRanks_) + " has already checked in", ErrorCode::ncclInternalError);
    }

    // Save the connection handle for that rank
		rankHandlesRoot[info.rank] = info.extHandleListenRoot;
		rankHandles[info.rank] = info.extHandleListen;

		++c;
    TRACE(NCCL_INIT, "Received connect from rank %d total %d/%d",  info.rank, c, nRanks_);
  } while (c < nRanks_);
  TRACE(NCCL_INIT, "COLLECTED ALL %d HANDLES", nRanks_);

  // Send the connect handle for the next rank in the AllGather ring
  for (int r=0; r<nRanks_; ++r) {
    int next = (r + 1) % nRanks_;
    Socket::tmpPackedSend(&rankHandlesRoot[r], &rankHandles[next], sizeof(SocketAddress));
  }
  TRACE(NCCL_INIT, "SENT OUT ALL %d HANDLES", nRanks_);

  TRACE(NCCL_INIT, "DONE");

}

void TcpBootstrap::Impl::createRoot() {
  rootThread_ = std::thread([this](){ 
    CHECKBACK(root());
    TRACE(NCCL_INIT, "FINE");
  });
}

void TcpBootstrap::Impl::initialize(InternalUniqueId *uniqueId) {
  uniqueId_ = *uniqueId;
  addr_ = getAddress(uniqueId);
  ExtInfo info = {rank_, nRanks_, addr_, addr_};
  listenSock_ = Socket::createListenSocket(&info.extHandleListen);

  // stagger connection times to avoid an overload of the root at very high rank counts
  if (nRanks_ > 128) {
    long msec = rank_;
    struct timespec tv;
    tv.tv_sec = msec / 1000;
    tv.tv_nsec = 1000000 * (msec % 1000);
    TRACE(NCCL_INIT, "rank %d delaying connection to root by %ld msec", rank_, msec);
    (void) nanosleep(&tv, NULL);
  }
  SocketAddress extHandleNext;
  {
    std::unique_ptr<Socket> extBstrapListenCommRoot = Socket::createListenSocket(&info.extHandleListenRoot);
    TRACE(NCCL_INIT, "Sending to root @@ %d", rank_);
    // send info on my listening socket to root
    Socket::tmpPackedSend(&uniqueId->addr, &info, sizeof(info));
    // get info on my "next" rank in the bootstrap ring from root
    Socket::tmpPackedRecv(extBstrapListenCommRoot.get(), &extHandleNext, sizeof(extHandleNext));
  }
  ringSendSock_ = Socket::connectAddress(&extHandleNext);
  // Accept the connect request from the previous rank in the AllGather ring
  ringRecvSock_ = listenSock_->accept();

  // AllGather all listen handlers
  peersAddr_[rank_] = addr_;
  allGather(peersAddr_.data(), sizeof(SocketAddress));

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank_, nRanks_);
}

void TcpBootstrap::Impl::allGather(void* allData, int size) {
  char *data = static_cast<char*>(allData);
  TRACE(NCCL_INIT, "rank %d nranks %d size %d", rank_, nRanks_, size);
  /* Simple ring based AllGather
   * At each step i receive data from (rank-i-1) from left
   * and send previous step's data from (rank-i) to right
   */
  for (int i=0; i<nRanks_-1; i++) {
    size_t rslice = (rank_ - i - 1 + nRanks_) % nRanks_;
    size_t sslice = (rank_ - i + nRanks_) % nRanks_;
    // Send slice to the right
    ringSendSock_->packedSend(data+sslice*size, size);
    // Recv slice from the left
    ringRecvSock_->packedRecv(data+rslice*size, size);
  }
  TRACE(NCCL_INIT, "rank %d nranks %d size %d - DONE", rank_, nRanks_, size);
}

void TcpBootstrap::Impl::send(void* data, int size, int peer) {
  std::unique_ptr<Socket> sock = Socket::connectAddress(&peersAddr_[peer]);
  sock->packedSend(&peer, sizeof(int));
  sock->packedSend(data, size);
}

void TcpBootstrap::Impl::recv(void* data, int size, int peer) {
  for (auto it = sockQueue_.begin(); it != sockQueue_.end(); ++it) {
    if (it->first == peer) {
      it->second->packedRecv(data, size);
      sockQueue_.erase(it);
      return;
    }
  }
  while (true) {
    std::unique_ptr<Socket> sock = listenSock_->accept();
    int rank;
    sock->packedRecv(&rank, sizeof(int));
    if (rank == peer) {
      sock->packedRecv(data, size);
      return;
    }
    sockQueue_.push_back(std::make_pair(rank, std::move(sock)));
  }
}

void TcpBootstrap::Impl::close() {
  listenSock_.reset(nullptr);
  ringSendSock_.reset(nullptr);
  ringRecvSock_.reset(nullptr);
  for (auto &pair : sockQueue_) {
    pair.second.reset(nullptr);
  }
  peersAddr_.clear();
  sockQueue_.clear();
}

UniqueId packUniqueId(InternalUniqueId uniqueId) {
  UniqueId res = {};
  memcpy(res.data(), &uniqueId, sizeof(InternalUniqueId));
  return res;
}

UniqueId TcpBootstrap::createUniqueId() { return packUniqueId(Impl::createUniqueId()); }
UniqueId TcpBootstrap::getUniqueId() const { return packUniqueId(pimpl_->getUniqueId()); }
TcpBootstrap::TcpBootstrap(int rank, int nRanks) { pimpl_ = std::make_unique<Impl>(rank, nRanks); }
TcpBootstrap::~TcpBootstrap() { pimpl_->close(); };
void TcpBootstrap::initialize(UniqueId *uniqueId) { pimpl_->initialize(reinterpret_cast<InternalUniqueId*>(uniqueId->data())); }
int TcpBootstrap::getRank() { return pimpl_->getRank(); }
int TcpBootstrap::getNranks() { return pimpl_->getNranks(); }
void TcpBootstrap::send(void* data, int size, int peer) { pimpl_->send(data, size, peer); }
void TcpBootstrap::recv(void* data, int size, int peer) { pimpl_->recv(data, size, peer); }
void TcpBootstrap::allGather(void* allData, int size) { pimpl_->allGather(allData, size); }

} // namespace ncclpp