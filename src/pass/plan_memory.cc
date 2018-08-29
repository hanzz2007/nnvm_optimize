/*!
 *  Copyright (c) 2016 by Contributors
 * \file plan_memory.cc
 * \brief Assign memory tag to each of the data entries.
 */
#include <nnvm/graph.h>
#include <nnvm/pass.h>
#include <nnvm/graph_attr_types.h>
#include <nnvm/op_attr_types.h>
#include <memory>
#include "./graph_algorithm.h"

namespace nnvm {
namespace pass {
namespace {

// qiuhan 计算图变量静态分配器
// 注意: 这个分配器只是根据计算图构建静态的存储分配策略
// 后面会根据这里构造的分配策略分配实际存储空间
// simple graph based allocator.
class GraphAllocator {
 public:
  // storage id equals integer.
  using StorageID = int;

  // bad storage id
  static const StorageID kBadStorageID = -1;
  // external storage id
  static const StorageID kExternalStorageID = -2;
  // dynamic storage id
  static const StorageID kDynamicStorageID = -3;

  // request a free storage
  StorageID Request(int dev_id, int dtype, TShape shape, uint32_t node_id) {
    if (shape.ndim() == 0) return kBadStorageID;
    // search memory block in [size / match_range_, size * match_range_)
    // TODO(tqchen) add size of the dtype, assume 4 bytes for now
    size_t size = shape.Size() * 4;
    if (match_range_ == 0) return this->Alloc(dev_id, size);
    auto begin = free_.lower_bound(size / match_range_); // qiuhan >= size / match_range_
    auto mid = free_.lower_bound(size);  // qiuhan >= size
    auto end = free_.upper_bound(size * match_range_); // qiuhan > size * match_range_
    // search for memory blocks larger than requested
    for (auto it = mid; it != end; ++it) {
      StorageEntry *e = it->second;
      // qiuhan 注意: 共享的存储空间必须位于同一个设备并且只由着色相同的节点使用
      // 如果着色不相同，有可能存在竞争冲突
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      // qiuhan 其实这一步应该是不需要的, e.max_bytes恒大于等于size
      e->max_bytes = std::max(size, e->max_bytes);
      // find a exact match, erase from map and return
      free_.erase(it);
      return e->id;
    }
    // then search for memory blocks smaller than requested space
    for (auto it = mid; it != begin;) {
      --it;
      StorageEntry *e = it->second;
      if (e->device_id != dev_id) continue;
      if (node_color_.size() != 0 &&
          node_color_[e->released_by_node] != node_color_[node_id]) continue;
      // Use exect matching strategy
      // qiuhan 这里下面这行代码才是必须的
      e->max_bytes = std::max(size, e->max_bytes); // qiuhan 扩大原来的小内存块
      // erase from map and return
      free_.erase(it);
      return e->id;
    }
    // cannot find anything return a new one.
    return this->Alloc(dev_id, size);
  }
  // release a memory space.
  void Release(StorageID id, uint32_t node_id) {
    CHECK_NE(id, kBadStorageID);
    // qiuhan 存储空间由外面分配，或者动态分配
    if (id == kExternalStorageID || id == kDynamicStorageID) return;
    StorageEntry *e = data_[id].get();
    e->released_by_node = node_id; // qiuhan 记录上一次使用的节点
    free_.insert({e->max_bytes, e});
  }

  // totoal number of bytes allocated
  size_t TotalAllocBytes() const {
    size_t total = 0;
    for (auto &p : data_) {
      total += p->max_bytes;
    }
    return total;
  }

  // constructor
  explicit GraphAllocator(const IndexedGraph* idx, const size_t match_range) : idx_(idx) {
    this->Init(match_range, dmlc::GetEnv("NNVM_EXEC_NUM_TEMP", 1));
  }

 private:
  // initialize the graph allocator
  void Init(const size_t match_range, const uint32_t num_match_color) {
    match_range_ = match_range;
    num_match_color_ = num_match_color;
    if (num_match_color_ > 1) {
      std::vector<uint32_t> importance(idx_->num_nodes(), 0);
      for (uint32_t nid = 0; nid < idx_->num_nodes(); ++nid) {
        if ((*idx_)[nid].source->is_variable()) continue; // qiuhan var节点内存不需要共享
        importance[nid] = 1;
      }
      // qiuhan 按官网文档的说法，以此沿最深的路径对节点进行着色
      // 同一条路径上的节点颜色相同，而内存共享操作只会在相同颜色的节点间进行，
      // 这样可以减少可能并发执行的不同的路径间的变量冲突，尽可能实现并行
      num_match_color_ = pass::ColorNodeGroup(
          *idx_, importance, num_match_color_, &node_color_);
    }
  }

  StorageID Alloc(int dev_id, size_t size) {
    StorageID id = static_cast<StorageID>(data_.size());
    std::unique_ptr<StorageEntry> ptr(new StorageEntry());
    ptr->id = id;
    ptr->device_id = dev_id;
    ptr->max_bytes = size;
    data_.emplace_back(std::move(ptr));
    return id;
  }
  // internal storage entry
  struct StorageEntry {
    // the id of the entry.
    StorageID id;
    // the device id of the storage.
    int device_id;
    // maximum size of storage requested.
    size_t max_bytes{0};
    // node index that released it last time
    uint32_t released_by_node{0};
  };
  // scale used for rough match
  size_t match_range_;
  // whether use color based match algorithm
  uint32_t num_match_color_{1};
  // the size of each dtype
  std::vector<size_t> dtype_size_dict_;
  // free list of storage entry
  std::multimap<size_t, StorageEntry*> free_;
  // all the storage resources available
  std::vector<std::unique_ptr<StorageEntry> > data_;
  // color of nodes in the graph, used for auxiliary policy making.
  std::vector<uint32_t> node_color_;
  // internal indexed graph
  const IndexedGraph* idx_;
};

class Chunk;
using ChunkPtr = std::shared_ptr<Chunk>;

#define DISABLE_COPY_AND_ASSIGN(NAME_) \
    private: \
        NAME_(const NAME_ &); \
        NAME_(const NAME_ &&); \
        NAME_ & operator=(const NAME_ &); \
        NAME_ & operator=(const NAME_ &&);

class ChunkSlice
{
public:
    ChunkSlice(const ChunkPtr& chunk,
        const uint32_t offset,
        const uint32_t size)
        : chunk_(chunk), offset_(offset_), size_(size)
    {
    }

    ChunkSlice() { }

    ChunkSlice slice(const uint32_t offset, const uint32_t size) {
        CHECK_LE(offset + offset_ + size, chunk_->size());
        return ChunkSlice(chunk_, offset_ + offset, size);
    }

    ChunkPtr source() const {
        return chunk_;
    }

    bool is_null() const {
        return chunk_ == nullptr;
    }

    uint32_t offset() const {
        return offset_;
    }

    uint32_t size() const {
        return size_;
    }

    ChunkSlice lower() const {
        return ChunkSlice(chunk_->root_parent(), offset_ + chunk_->root_offset(), size_);
    }

    friend bool operator< (const ChunkSlice& lhs, const ChunkSlice& rhs) {
        return lhs.source() < rhs.source() || lhs.size() < rhs.size();
    }

private:
    ChunkPtr chunk_;
    uint32_t offset_{ 0 };
    uint32_t size_{ 0 };
};


class Chunk
    : std::enable_shared_from_this<Chunk>
{
    DISABLE_COPY_AND_ASSIGN(Chunk)

public:
    enum class ExpandDirection {
        kExpandHead,
        kExpandTail
    };

    static ChunkPtr make(const uint32_t size) {
        return std::make_shared<Chunk>(size);
    }

    uint32_t size() const {
        return size_;
    }

    ChunkSlice slice(const uint32_t offset, const uint32_t size) {
        CHECK_LE(offset + size, size_);
        return ChunkSlice(shared_from_this(), offset, size);
    }

    ChunkPtr expand(const uint32_t new_size, const ExpandDirection direct)
    {
        CHECK(!parent_);
        CHECK_EQ(offset_, 0);
        CHECK_GE(new_size, size_);

        if (direct == ExpandDirection::kExpandHead) {
            offset_ = new_size - size_;
        }
        parent_ = make(new_size);

        return parent_;
    }

    ChunkPtr embed_in(const ChunkPtr& parent, const uint32_t offset)
    {
        CHECK(!parent_);
        CHECK_LE(offset_ + size_, parent->size());

        parent_ = parent;
        offset_ = offset;

        return parent_;
    }

    ChunkPtr parent() const {
        return parent_;
    }

    // 注意: root_parent有可能是它自己
    ChunkPtr root_parent() const 
    {
        auto gp = shared_from_this();
        // 这里不用递归防止爆栈
        while (gp->parent()) {
            gp = gp->parent();
        }
        return gp;
//         return std::dynamic_pointer_cast<ChunkPtr>(gp);
    }

    uint32_t offset() const {
        return offset_;
    }

    uint32_t root_offset() const
    {
        uint32_t offset = offset_;
        ChunkPtr gp = parent_;
        // 这里不用递归防止爆栈
        while (gp) {
            offset += gp->offset();
            gp = gp->parent();
        }
        return offset;
    }

private:
    Chunk(const uint32_t size) 
        : size_(size) 
    {
    }

    uint32_t size_{ 0 };
    uint32_t offset_{ 0 };
    ChunkPtr parent_;
};

size_t PlanChunk(const Graph& ret, const IndexedGraph& idx,
    const std::pair<uint32_t, uint32_t>& node_range,
    std::vector<ChunkSlice>* storage_ptr,
    std::vector<int>* storage_inplace_index_ptr,
    const std::vector<uint32_t>& entry_ref_count) 
{
    static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");
    static auto& finplace_identity = Op::GetAttr<FInplaceIdentity>("FInplaceIdentity");

    // Get reference
    auto &storage = *storage_ptr;
    auto &storage_inplace_index = *storage_inplace_index_ptr;

    // Get attributes from the graph
    const ShapeVector& shape_vec = ret.GetAttr<ShapeVector>("shape");
    const DTypeVector& dtype_vec = ret.GetAttr<DTypeVector>("dtype");
    const DeviceVector* device_vec = nullptr;

    if (ret.attrs.count("device") != 0) {
        device_vec = &(ret.GetAttr<DeviceVector>("device"));
    }

    size_t num_not_allocated = 0;
    std::unordered_map<ChunkSlice, int32_t> storage_ref_count;
    std::unordered_map<ChunkPtr, int32_t> chunk_ref_count;

    for (uint32_t nid = node_range.first; nid < node_range.second; ++nid)
    {
        const auto& inode = idx[nid];
        if (inode.source->is_variable()) continue;
        // check inplace option
        if (finplace_option.count(inode.source->op()) != 0) {
            auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
            std::vector<bool> identity;
            if (finplace_identity.count(inode.source->op()) != 0) {
                identity = finplace_identity[inode.source->op()](inode.source->attrs);
                CHECK_EQ(identity.size(), inplace_pairs.size())
                    << "FInplaceOption and FInplaceIdentity returned vectors of different "
                    << "size for operator " << inode.source->op()->name;
            }
            else {
                identity = std::vector<bool>(inplace_pairs.size(), false);
            }

            std::vector<bool> taken(inode.inputs.size(), false);

            for (size_t ipair = 0; ipair < inplace_pairs.size(); ++ipair) {
                const auto& kv = inplace_pairs[ipair];
                uint32_t eid_out = idx.entry_id(nid, kv.second);
                uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
                const auto& sid_out = storage[eid_out];
                const auto& sid_in = storage[eid_in];

                if (taken[kv.first] == false &&
                    sid_out.is_null() &&
                    !sid_in.is_null() &&
                    // 注意这里, 这里根据root_chunk的引用数判断， 对于inplace而言不允许别人需要引用chunk的时候，做inplace修改 
                    (chunk_ref_count[sid_in.source()->root_parent()] == 1 || identity[ipair]) &&
                    entry_ref_count[eid_out] > 0 &&
                    shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
                    dtype_vec[eid_out] == dtype_vec[eid_in])
                {
                    // inplace optimization
                    taken[kv.first] = true;
                    storage[eid_out] = sid_in;
                    storage_ref_count[sid_in] += entry_ref_count[eid_out];
                    chunk_ref_count[sid_in.source()->root_parent()] += entry_ref_count[eid_out];
                    storage_inplace_index[eid_out] = kv.first;
                }
            }
        }
        else if (true) // concat
        {
            CHECK_EQ(inode.source->num_outputs(), 1);
            uint32_t eid_out = idx.entry_id(nid, 0);

            const uint32_t out_size = shape_vec[eid_out].Size();
            ChunkSlice sid_out = Chunk::make(out_size)->slice(0, out_size);

            uint32_t offset = 0;
            std::unordered_set<ChunkPtr> in_chunks;

            for (size_t i = 0; i < inode.source->num_inputs(); ++i)
            {
                uint32_t eid_in = idx.entry_id(inode.source->inputs[i]);
                ChunkSlice sid_in = storage[eid_in];
                CHECK(!sid_in.is_null());

                const uint32_t in_size = shape_vec[eid_in].Size();
                ChunkSlice sid_in_lower = sid_in.lower();

                if (in_chunks.count(sid_in_lower.source()) < 1)
                {
                    ChunkSlice sid_out_lower = sid_out.lower();

                    if (sid_in_lower.offset() == 0 &&
                        sid_in_lower.size() == sid_in_lower.source()->size())
                    {
                        sid_in_lower.source()->embed_in(sid_out.source(), offset);

                        chunk_ref_count[sid_out.source()->root_parent()] += chunk_ref_count[sid_in_lower.source()];
                        chunk_ref_count[sid_in_lower.source()] = 0;
                    }
                    else if (i == 0 &&
                        sid_in_lower.offset() + sid_in_lower.size() == sid_in_lower.source()->size() &&
                        sid_out_lower.offset() == 0)
                    {
                        uint32_t new_size = sid_out_lower.source()->size() + sid_in_lower.source()->size() - in_size;
                        ChunkPtr shared_chunk = sid_out_lower.source()->expand(new_size, Chunk::ExpandDirection::kExpandHead);
                        sid_in_lower.source()->embed_in(shared_chunk, 0);

                        chunk_ref_count[shared_chunk] = chunk_ref_count[sid_out_lower.source()] + chunk_ref_count[sid_in_lower.source()];
                        chunk_ref_count[sid_out_lower.source()] = chunk_ref_count[sid_in_lower.source()] = 0;
                    }
                    else if (i + 1 == inode.source->num_inputs() &&
                        sid_in_lower.offset() == 0 &&
                        sid_out_lower.offset() + sid_out_lower.size() == sid_out_lower.source()->size())
                    {
                        uint32_t new_size = sid_out_lower.source()->size() + sid_in_lower.source()->size() - in_size;
                        ChunkPtr shared_chunk = sid_out_lower.source()->expand(new_size, Chunk::ExpandDirection::kExpandTail);
                        sid_in_lower.source()->embed_in(shared_chunk, new_size - sid_in_lower.source()->size());

                        chunk_ref_count[shared_chunk] = chunk_ref_count[sid_out_lower.source()] + chunk_ref_count[sid_in_lower.source()];
                        chunk_ref_count[sid_out_lower.source()] = chunk_ref_count[sid_in_lower.source()] = 0;
                    }
                    else {
                        // Share nothing
                        LOG(INFO) << "Op: " << inode.source->attrs.name <<  " no share input: " << i;
                    }
                }

                offset += shape_vec[eid_in].Size();
            }

            storage[eid_out] = sid_out;
            storage_ref_count[sid_out] = entry_ref_count[eid_out];
            chunk_ref_count[sid_out.source()->root_parent()] += entry_ref_count[eid_out];
        }

        // normal allocation
        const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;
        // sort output nodes based on size before allocating output
        std::multimap<size_t, uint32_t> eids;
        for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
            uint32_t eid = idx.entry_id(nid, index);
            // only request memory for kBadStorageID
            if (storage[eid].is_null()) {
                auto &eshape = shape_vec[eid];
                size_t esize = 0;
                if (eshape.ndim() != 0) esize = eshape.Size();
                eids.insert(std::make_pair(esize, eid));
            }
        }
        for (auto rit = eids.rbegin(); rit != eids.rend(); ++rit) {
            uint32_t eid = rit->second;
//             auto sid = allocator->Request(dev_id, dtype_vec[eid], shape_vec[eid], nid);
            auto sid = Chunk::make(shape_vec[eid].Size())->slice(0, shape_vec[eid].Size());
            storage_ref_count[sid] = entry_ref_count[eid];
            chunk_ref_count[sid.source()->root_parent()] = entry_ref_count[eid];
            storage[eid] = sid;
        }

        // check if certain inputs is ignored.
        static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
        std::vector<uint32_t> ignore_inputs;
        if (fignore_inputs.count(inode.source->op()) != 0) {
            ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
            std::sort(ignore_inputs.begin(), ignore_inputs.end());
        }
        // then free inputs
        for (size_t i = 0; i < inode.inputs.size(); ++i) {
            // ref counter of ignored input is already decreased.
            if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
            const auto& e = inode.inputs[i];
            uint32_t eid = idx.entry_id(e);
            auto sid = storage[eid];
            // storage_ref_count == 0 means it is taken by inplace op
            if (sid.is_null()) continue;
            // if we decrease it to zero, means we are ready to relase
            --storage_ref_count[sid];
            --chunk_ref_count[sid.source()->root_parent()];
            if (storage_ref_count[sid] == 0) {
//                 allocator->Release(sid, nid);
            }
        }

        // check if there are outputs that can be freeded immediately
        // these output are not referenced by any operator.
        for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
            uint32_t eid = idx.entry_id(nid, index);
            auto sid = storage[eid];
            if (!sid.is_null() && storage_ref_count[sid] == 0) {
//                 allocator->Release(sid, nid);
                // use -2 to indicate that the node was never touched.
                storage_inplace_index[eid] = -2; // qiuhan 注意这里！！ 该输出完全不会被用到
                                                 // qiuhan 该节点的输出的opreqtype为NULL
            }
            if (storage[eid].is_null()) {
                ++num_not_allocated; // qiuhan 多少个节点没有被分配内存
            }
        }
    }
    return num_not_allocated;
}


/*
 * Internal method to perform the memory allocation for a graph
 * */
size_t AllocMemory(const Graph& ret, const IndexedGraph& idx,
                   const std::pair<uint32_t, uint32_t>& node_range,
                   StorageVector* storage_ptr,
                   std::vector<int>* storage_inplace_index_ptr,
                   const std::vector<uint32_t>& entry_ref_count,
                   GraphAllocator* allocator) {
  static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");
  static auto& finplace_identity = Op::GetAttr<FInplaceIdentity>("FInplaceIdentity");

  // Get reference
  auto &storage = *storage_ptr;
  auto &storage_inplace_index = *storage_inplace_index_ptr;

  // Get attributes from the graph
  const ShapeVector& shape_vec = ret.GetAttr<ShapeVector>("shape");
  const DTypeVector& dtype_vec = ret.GetAttr<DTypeVector>("dtype");
  const DeviceVector* device_vec = nullptr;

  if (ret.attrs.count("device") != 0) {
    device_vec = &(ret.GetAttr<DeviceVector>("device"));
  }
  size_t num_not_allocated = 0;
  // qiuhan storage_ref_count就是官网文档中示例图中分配过程中不断变化的存储引用计数
  // 就是说当前该内存中的内容还会被多少个节点引用，所以当>0时,该内存暂不能被复用
  std::vector<GraphAllocator::StorageID> storage_ref_count(idx.num_node_entries(), 0);

  // qiuhan 模拟执行计算图，规划存储的分配和释放路径，以此构建存储分配策略
  for (uint32_t nid = node_range.first; nid < node_range.second; ++nid) {
    const auto& inode = idx[nid];
    if (inode.source->is_variable()) continue;
    // check inplace option
    if (finplace_option.count(inode.source->op()) != 0) {
      auto inplace_pairs = finplace_option[inode.source->op()](inode.source->attrs);
      std::vector<bool> identity;
      if (finplace_identity.count(inode.source->op()) != 0) {
        identity = finplace_identity[inode.source->op()](inode.source->attrs);
        CHECK_EQ(identity.size(), inplace_pairs.size())
            << "FInplaceOption and FInplaceIdentity returned vectors of different "
            << "size for operator " << inode.source->op()->name;
      } else {
        identity = std::vector<bool>(inplace_pairs.size(), false);
      }

      // qiuhan taken: 防止某个节点的多个输出可能都要在同一个input做inplace优化
      // 如果输入已经被拿做inplace优化了，对应的taken元素就为true，下一个能做inplace
      // 的节点就不能真的做inplace了
      std::vector<bool> taken(inode.inputs.size(), false);

      for (size_t ipair = 0; ipair < inplace_pairs.size(); ++ipair) {
        const auto& kv = inplace_pairs[ipair];
        uint32_t eid_out = idx.entry_id(nid, kv.second);
        uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
        auto sid_out = storage[eid_out];
        auto sid_in = storage[eid_in];
        // qiuhan 判断是否满足inplace优化条件
        if (taken[kv.first] == false &&
            sid_out == GraphAllocator::kBadStorageID &&
            sid_in >= 0 &&
            // qiuhan 1. inplace op && refcount必须为1
            // 2. 恒等inplace op , refcount无要求
            (storage_ref_count[sid_in] == 1 || identity[ipair]) &&
            // qiuhan input不是var类型节点的输出. var节点的输出对应的entry_ref_count为0
            // 因为var节点的存储是不能够做共享的, 还有种情况就是该输出没有其他节点引用
            entry_ref_count[eid_out] > 0 &&
            // qiuhan inplace共享的存储的大小和类型都要一样
            shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
            dtype_vec[eid_out] == dtype_vec[eid_in]) 
        {
          // inplace optimization
          taken[kv.first] = true; // qiuhan 该input已经做了inplace共享了
          storage[eid_out] = sid_in; // input和output共享同一个storage id
          // Reuse storage for output and add ref count of output
          // to storage. This will get substracted later in free
          // input section.
          // qiuhan 当前该内存种的内容还有多少个节点需要访问，所以不能拿来内存共享
          storage_ref_count[sid_in] += entry_ref_count[eid_out]; 
          storage_inplace_index[eid_out] = kv.first;
        }
      }
    }

    // normal allocation
    const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;
    // sort output nodes based on size before allocating output
    std::multimap<size_t, uint32_t> eids; // qiuhan 当前节点的哪些输出需要分配内存
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      // only request memory for kBadStorageID
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        auto &eshape = shape_vec[eid];
        size_t esize = 0;
        if (eshape.ndim() != 0) esize = eshape.Size();
        eids.insert(std::make_pair(esize, eid));
      }
    }
    for (auto rit = eids.rbegin(); rit != eids.rend(); ++rit) {
        uint32_t eid = rit->second;
        auto sid = allocator->Request(dev_id, dtype_vec[eid], shape_vec[eid], nid);
        storage_ref_count[sid] = entry_ref_count[eid];
        storage[eid] = sid;
    }

    // qiuhan 当前节点已经‘执行'完毕，input的数据已经不需要了
    // 输入的存储引用计数要-1
    // check if certain inputs is ignored.
    static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
    std::vector<uint32_t> ignore_inputs;
    if (fignore_inputs.count(inode.source->op()) != 0) {
      ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
      std::sort(ignore_inputs.begin(), ignore_inputs.end());
    }
    // then free inputs
    for (size_t i = 0; i < inode.inputs.size(); ++i) {
      // ref counter of ignored input is already decreased.
      if (std::binary_search(ignore_inputs.begin(), ignore_inputs.end(), i)) continue;
      const auto& e = inode.inputs[i];
      uint32_t eid = idx.entry_id(e);
      auto sid = storage[eid];
      // storage_ref_count == 0 means it is taken by inplace op
      if (sid < 0) continue;
      // if we decrease it to zero, means we are ready to relase
      --storage_ref_count[sid];
      if (storage_ref_count[sid] == 0) {
          // qiuhan 当前没有其他节点引用这个数据了,可以插入到空闲列表等待复用
        allocator->Release(sid, nid); 
      }
    }

    // qiuhan 检查当前节点的输出是否有被其他节点用到
    // 如果没有就释放存储到空闲链表，可能会被下次复用
    // check if there are outputs that can be freeded immediately
    // these output are not referenced by any operator.
    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
      uint32_t eid = idx.entry_id(nid, index);
      auto sid = storage[eid];
      if (sid >= 0 && storage_ref_count[sid] == 0) {
        allocator->Release(sid, nid);
        // use -2 to indicate that the node was never touched.
        storage_inplace_index[eid] = -2; // qiuhan 注意这里！！ 该输出完全不会被用到
        // qiuhan 该节点的输出的opreqtype为NULL
      }
      if (storage[eid] == GraphAllocator::kBadStorageID) {
        ++num_not_allocated; // qiuhan 多少个节点没有被分配内存
      }
    }
  }
  return num_not_allocated;
}


// function to plan memory
Graph PlanMemory(Graph ret) {
  // setup ref counter
  const IndexedGraph& idx = ret.indexed_graph();
  static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
  std::pair<uint32_t, uint32_t> node_range = {0, idx.num_nodes()};
  if (ret.attrs.count("node_range")) {
    node_range = ret.MoveCopyAttr<std::pair<uint32_t, uint32_t> >("node_range");
  }

  // qiuhan 第一步: 计算每个存储变量的引用计数(就是会被多少个节点作为输入)
  // reference counter of each node
  std::vector<uint32_t> ref_count;
  // step 1: initialize reference count
  if (ret.attrs.count("ref_count") != 0) {
    ref_count = ret.MoveCopyAttr<std::vector<uint32_t> >("ref_count");
  } else {
    ref_count.resize(idx.num_node_entries(), 0);
    for (uint32_t nid = 0; nid < idx.num_nodes(); ++nid) {
      const auto& inode = idx[nid];
      if (inode.source->is_variable()) continue;
      for (const auto& e : inode.inputs) {
        ++ref_count[idx.entry_id(e)];
      }
      // no dataflow dependency is needed for those are ignored.
      // revoke the dependency counter.
      if (fignore_inputs.count(inode.source->op()) != 0) {
        auto ignore_inputs = fignore_inputs[inode.source->op()](inode.source->attrs);
        for (uint32_t i : ignore_inputs) {
          --ref_count[idx.entry_id(inode.inputs[i])];
        }
      }
    }

    // qiuhan 网络的输出可能没有其他节点作为输入来引用，但是也要+1
    for (const auto& e : idx.outputs()) {
      ++ref_count[idx.entry_id(e)];
    }
  }

  // qiuhan 第二步: 分配内存
  // step 2: allocate memory.
  StorageVector storage;
  if (ret.attrs.count("storage") != 0) {
    storage = ret.MoveCopyAttr<StorageVector>("storage");
  } else {
    storage.resize(idx.num_node_entries(), -1);
  }


  // qiuhan 注意: match_range
  // Allocator中的free_list是个按空间大小升序排列，如果要从free_list中分配size大小
  // 的空间，就从[size/match_range, size*match_range]中找一个最合适的空闲存储块，
  // 满足条件的match_range越小，空间浪费就越小，所以下面的for循环就是为了能找到最合适
  // 的match_range
  // Search the best NNVM_EXEC_MATCH_RANGE parameter. This is turned off by default
  size_t min_allocated_bytes = -1;
  size_t max_match_range = dmlc::GetEnv("NNVM_EXEC_MATCH_RANGE", 16);
  size_t min_match_range =
         dmlc::GetEnv("NNVM_AUTO_SEARCH_MATCH_RANGE", false) ? 1 : max_match_range;
  for (size_t match_range = min_match_range; match_range <= max_match_range; match_range *= 2) {
    // Make a copy of related fields
    StorageVector storage_vec(storage);
    std::vector<int> storage_inplace_index(idx.num_node_entries(), -1);

    // the allocator
    GraphAllocator allocator(&idx, match_range);

    // number of entries that are not statically allocated.
    size_t storage_num_not_allocated =
      AllocMemory(ret, idx, node_range, &storage_vec, &storage_inplace_index,
                  ref_count, &allocator);
    size_t storage_allocated_bytes = allocator.TotalAllocBytes();
    // qiuhan 这么做有问题吧???
    // 前提是每次循环storage_num_not_allocated都不能大于上一次循环
    // 否则可能出现上一次循环分配都成功，下一次循环却分配失败了的问题
    // Choose the plan which leads to minimal memory usage
    if (min_allocated_bytes > storage_allocated_bytes) {
      ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
      ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
      ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(storage_allocated_bytes);
      ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(storage_num_not_allocated);
      min_allocated_bytes = storage_allocated_bytes;
    }

    if (max_match_range == 0) {
      break;
    }
  }
  return ret;
}

NNVM_REGISTER_PASS(PlanMemory)
.describe("Plan the memory allocation of each node entries.")
.set_body(PlanMemory)
.set_change_graph(false)
.depend_graph_attr("dtype")
.depend_graph_attr("shape")
.provide_graph_attr("storage_id")
.provide_graph_attr("storage_inplace_index");

}  // namespace
}  // namespace pass
}  // namespace nnvm
