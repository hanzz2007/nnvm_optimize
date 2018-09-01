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
#include "mshadow/base.h"
#include "chunk.h"

namespace nnvm {
    namespace pass {
        namespace {
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
                    return Request(dev_id, size, node_id);
                }

                StorageID Request(int dev_id, size_t size, uint32_t node_id)
                {
                    if (match_range_ == 0) return this->Alloc(dev_id, size);
                    auto begin = free_.lower_bound(size / match_range_); // qiuhan >= size / match_range_
                    auto mid = free_.lower_bound(size);  // qiuhan >= size
                    auto end = free_.upper_bound(size * match_range_); // qiuhan > size * match_range_
                    // search for memory blocks larger than requested
                    for (auto it = mid; it != end; ++it) {
                        StorageEntry *e = it->second;
                        if (e->device_id != dev_id) continue;
                        if (node_color_.size() != 0 &&
                            node_color_[e->released_by_node] != node_color_[node_id]) continue;
                        // Use exect matching strategy
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
                        e->max_bytes = std::max(size, e->max_bytes);
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
                    if (id == kExternalStorageID || id == kDynamicStorageID) return;
                    StorageEntry *e = data_[id].get();
                    e->released_by_node = node_id;
                    free_.insert({ e->max_bytes, e });
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
                            if ((*idx_)[nid].source->is_variable()) continue;
                            importance[nid] = 1;
                        }
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
                    size_t max_bytes{ 0 };
                    // node index that released it last time
                    uint32_t released_by_node{ 0 };
                };
                // scale used for rough match
                size_t match_range_;
                // whether use color based match algorithm
                uint32_t num_match_color_{ 1 };
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

            struct InputSlice
            {
                ChunkSlice sid;
                size_t offset;
                size_t beg;
                size_t end;
            };

            size_t PlanChunk(const Graph& ret, const IndexedGraph& idx,
                const std::pair<uint32_t, uint32_t>& node_range,
                std::vector<ChunkSlice>* storage_ptr,
                std::vector<int>* storage_inplace_index_ptr,
                const std::vector<uint32_t>& entry_ref_count,
                std::unordered_map<ChunkPtr, int32_t>* chunk_ref_count_ptr)
            {
                static auto& finplace_option = Op::GetAttr<FInplaceOption>("FInplaceOption");
                static auto& finplace_identity = Op::GetAttr<FInplaceIdentity>("FInplaceIdentity");
                static auto concat_op = Op::Get("Concat");
                static auto slice_op = Op::Get("slice");
                static auto& fsplit_memory = Op::GetAttr<FMemorySplit>("FMemorySplit");
                static auto& fis_memory_fusable = Op::GetAttr<FIsMemoryFusable>("FIsMemoryFusable");

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

                std::map<ChunkSlice, int32_t> storage_ref_count;
                std::unordered_map<ChunkPtr, int32_t>& chunk_ref_count = *chunk_ref_count_ptr;

                for (uint32_t nid = node_range.first; nid < node_range.second; ++nid)
                {
                    const auto& inode = idx[nid];
                    if (inode.source->is_variable()) continue;
                    // check inplace option
                    if (finplace_option.count(inode.source->op()) != 0) 
                    {
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
                    
                    if (inode.source->op() == concat_op && 
                        fis_memory_fusable[inode.source->op()](
                            inode.source->attrs, shape_vec[idx.entry_id(nid, 0)])) // concat
                    {
                        CHECK_EQ(inode.source->num_outputs(), 1);
                       
                        std::vector<InputSlice> cand_input_chunks; // valid candidate input chunks (may be merged)
                        {
                            size_t offset = 0;
                            InputSlice prev_input;  // (merged_input, [beg_in, end_in])

                            for (size_t i = 0; i < inode.source->num_inputs(); ++i)
                            {
                                uint32_t eid_in = idx.entry_id(inode.source->inputs[i]);
                                const size_t in_size = shape_vec[eid_in].Size() * mshadow::mshadow_sizeof(dtype_vec[eid_in]);

                                ChunkSlice sid_in = storage[eid_in];
                                ChunkSlice sid_in_lower = sid_in.lower();

                                if (!prev_input.sid.is_null())
                                {
                                    ChunkSlice prev_sid_in_lower = prev_input.sid.lower();
                                    if (prev_sid_in_lower.source() == sid_in_lower.source() &&
                                        prev_sid_in_lower.offset() + prev_sid_in_lower.size() == sid_in_lower.offset())
                                    {
                                        // The two inputs are continous, can be merged
                                        prev_input.sid = prev_sid_in_lower.source()->slice(
                                            prev_sid_in_lower.offset(), prev_sid_in_lower.size() + sid_in_lower.size());
                                        prev_input.end = i;
                                    }
                                    else
                                    {
                                        // Prev inputs can be overlapped in Left side, or can be whole embeded
                                        if ((prev_sid_in_lower.offset() + prev_sid_in_lower.size() == prev_sid_in_lower.source()->size() &&
                                                prev_input.beg == 0)
                                            || (prev_sid_in_lower.size() == prev_sid_in_lower.source()->size()))
                                        {
                                            InputSlice new_cand = prev_input;
                                            new_cand.sid = prev_sid_in_lower;
                                            cand_input_chunks.push_back(new_cand);
                                        }

                                        prev_input = { sid_in_lower, offset, i, i};
                                    }
                                }
                                else {
                                    prev_input = { sid_in_lower, offset, i, i};
                                }

                                // Check the last
                                if (i + 1 == inode.source->num_inputs() && !prev_input.sid.is_null())
                                {
                                    CHECK_EQ(prev_input.end, i);
                                    // Last inputs can be overlapped in Right side, 
                                    // or out can be fully contained in input,
                                    // or input can be whole embeded in out, 
                                    ChunkSlice prev_sid_in_lower = prev_input.sid.lower();
                                    if ((prev_sid_in_lower.offset() == 0)
                                        || (prev_input.beg == 0)
                                        || (prev_sid_in_lower.size() == prev_sid_in_lower.source()->size()))
                                    {
                                        InputSlice new_cand = prev_input;
                                        new_cand.sid = prev_sid_in_lower;
                                        cand_input_chunks.push_back(new_cand);
                                    }
                                }

                                offset += in_size;
                            }

                            // We should sort to make the best choice
                            std::sort(cand_input_chunks.begin(), cand_input_chunks.end(),
                                [](const InputSlice& lhs, const InputSlice& rhs) {
                                return lhs.sid.size() > rhs.sid.size();
                            });
                        }

                        const uint32_t eid_out = idx.entry_id(nid, 0);
                        const size_t out_size = shape_vec[eid_out].Size() * mshadow::mshadow_sizeof(dtype_vec[eid_out]);
                        ChunkSlice sid_out = Chunk::make(out_size)->slice(0, out_size);
                        std::unordered_set<ChunkPtr> in_chunks;
                        auto fin_same_chunk = [](const std::unordered_set<ChunkPtr>& in_chunks, ChunkPtr chunk)
                        {
                            if (in_chunks.count(chunk) > 0) {
                                return true;
                            }
                           for (auto ck : in_chunks) {
                               ck = ck->parent();
                               while (ck && ck != chunk) {
                                   ck = ck->parent();
                               }
                               if (ck) {
                                   return true;
                               }
                           }
                           return false;
                        };

                        for (size_t i = 0; i < cand_input_chunks.size(); ++i)
                        {
                            const InputSlice in_ck = cand_input_chunks[i];
                            const size_t in_size = in_ck.sid.size();
                            const size_t offset = in_ck.offset;
                            const size_t beg = in_ck.beg, end = in_ck.end;
                            ChunkSlice sid_in_lower = in_ck.sid.lower();

                            // avoid non-cont same chunk inputs to be wrong fused into one out
                            if (!fin_same_chunk(in_chunks, sid_in_lower.source()))
                            {
                                ChunkSlice sid_out_lower = sid_out.lower();

                                if (sid_in_lower.offset() == 0 &&
                                    sid_in_lower.size() == sid_in_lower.source()->size())
                                {
                                    // input fully embeded in output
                                    sid_in_lower.source()->embed_in(sid_out.source(), offset);

                                    chunk_ref_count[sid_out.source()->root_parent()] += chunk_ref_count[sid_in_lower.source()];
                                    chunk_ref_count[sid_in_lower.source()] = 0;
                                    in_chunks.insert(sid_in_lower.source());
                                }
                                else if (beg == 0 &&
                                    sid_in_lower.offset() + sid_in_lower.size() == sid_in_lower.source()->size() &&
                                    sid_out_lower.offset() == 0)
                                {
                                    // 
//                                     CHECK_EQ(sid_out_lower.offset(), 0);
                                    size_t new_size = sid_out_lower.source()->size() + sid_in_lower.source()->size() - in_size;
                                    ChunkPtr shared_chunk = sid_out_lower.source()->expand(new_size, Chunk::ExpandDirection::kExpandHead);
                                    sid_in_lower.source()->embed_in(shared_chunk, 0);

                                    chunk_ref_count[shared_chunk] = chunk_ref_count[sid_out_lower.source()] + chunk_ref_count[sid_in_lower.source()];
                                    chunk_ref_count[sid_out_lower.source()] = chunk_ref_count[sid_in_lower.source()] = 0;
                                    in_chunks.insert(sid_in_lower.source());
                                }
                                else if (end + 1 == inode.source->num_inputs() &&
                                    sid_in_lower.offset() == 0 &&
                                    sid_out_lower.offset() + sid_out_lower.size() == sid_out_lower.source()->size())
                                {
                                    size_t new_size = sid_out_lower.source()->size() + sid_in_lower.source()->size() - in_size;
                                    ChunkPtr shared_chunk = sid_out_lower.source()->expand(new_size, Chunk::ExpandDirection::kExpandTail);
                                    sid_in_lower.source()->embed_in(shared_chunk, new_size - sid_in_lower.source()->size());

                                    chunk_ref_count[shared_chunk] = chunk_ref_count[sid_out_lower.source()] + chunk_ref_count[sid_in_lower.source()];
                                    chunk_ref_count[sid_out_lower.source()] = chunk_ref_count[sid_in_lower.source()] = 0;
                                    in_chunks.insert(sid_in_lower.source());
                                }
                                else if (end - beg + 1 == inode.source->num_inputs())
                                {
                                    CHECK_EQ(cand_input_chunks.size(), 1);
                                    // whole outputs share the slice of input chunk
                                    sid_out_lower.source()->embed_in(sid_in_lower.source(), sid_in_lower.offset());
                                    CHECK_EQ(chunk_ref_count[sid_out_lower.source()], 0);
                                    in_chunks.insert(sid_in_lower.source());
//                                     chunk_ref_count[sid_in_lower.source()] += chunk_ref_count[sid_out_lower.source()];
//                                     chunk_ref_count[sid_out_lower.source()] = 0;
                                }
                                else {
                                    // Share nothing
                                    LOG(INFO) << "Op: " << inode.source->attrs.name << " no share input: " << beg << "-" << end;
                                }
                            }

                        }

                        storage[eid_out] = sid_out;
                        storage_ref_count[sid_out] = entry_ref_count[eid_out];
                        chunk_ref_count[sid_out.source()->root_parent()] += entry_ref_count[eid_out];
                    }
                    
                    
                    if (fsplit_memory.count(inode.source->op()))
                    {
                        CHECK_EQ(inode.inputs.size(), 1);
//                          CHECK_EQ(inode.source->num_outputs(), 1);

                        const uint32_t eid_in = idx.entry_id(inode.source->inputs[0]);
                        auto sid_in = storage[eid_in];
                        if (sid_in.is_null()) continue;

                        std::vector<nnvm::TShape> out_shape_vec;
                        for (size_t i = 0; i < inode.source->num_outputs(); ++i) {
                            out_shape_vec.push_back(shape_vec[idx.entry_id(nid, i)]);
                        }
                        auto split_memory = fsplit_memory[inode.source->op()];
                        const auto offset_vec = split_memory(inode.source->attrs, { shape_vec[eid_in] }, out_shape_vec);
                        CHECK_EQ(offset_vec.size(), out_shape_vec.size());

                        for (size_t i = 0; i < offset_vec.size(); ++i) {
                            if (offset_vec[i] < 0) continue;
                            const uint32_t eid_out = idx.entry_id(nid, i);
                            size_t dtype_size = mshadow::mshadow_sizeof(dtype_vec[idx.entry_id(nid, i)]);
                            const size_t offset = static_cast<size_t>(offset_vec[i]) * dtype_size;
                            auto sid_out = sid_in.slice(offset, shape_vec[eid_out].Size() * dtype_size);
                            storage[eid_out] = sid_out;
                            storage_ref_count[sid_out] = entry_ref_count[eid_out];
                            chunk_ref_count[sid_in.source()->root_parent()] += entry_ref_count[eid_out];
                        }
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
                        const auto esize = shape_vec[eid].Size() * mshadow::mshadow_sizeof(dtype_vec[eid]);
                        auto sid = Chunk::make(esize)->slice(0, esize);
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
                std::vector<GraphAllocator::StorageID> storage_ref_count(idx.num_node_entries(), 0);

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
                        }
                        else {
                            identity = std::vector<bool>(inplace_pairs.size(), false);
                        }
                        std::vector<bool> taken(inode.inputs.size(), false);
                        for (size_t ipair = 0; ipair < inplace_pairs.size(); ++ipair) {
                            const auto& kv = inplace_pairs[ipair];
                            uint32_t eid_out = idx.entry_id(nid, kv.second);
                            uint32_t eid_in = idx.entry_id(inode.inputs[kv.first]);
                            auto sid_out = storage[eid_out];
                            auto sid_in = storage[eid_in];
                            if (taken[kv.first] == false &&
                                sid_out == GraphAllocator::kBadStorageID &&
                                sid_in >= 0 &&
                                (storage_ref_count[sid_in] == 1 || identity[ipair]) &&
                                entry_ref_count[eid_out] > 0 &&
                                shape_vec[eid_out].Size() == shape_vec[eid_in].Size() &&
                                dtype_vec[eid_out] == dtype_vec[eid_in]) {
                                // inplace optimization
                                taken[kv.first] = true;
                                storage[eid_out] = sid_in;
                                // Reuse storage for output and add ref count of output
                                // to storage. This will get substracted later in free
                                // input section.
                                storage_ref_count[sid_in] += entry_ref_count[eid_out];
                                storage_inplace_index[eid_out] = kv.first;
                            }
                        }
                    }
                    // normal allocation
                    const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;
                    // sort output nodes based on size before allocating output
                    std::multimap<size_t, uint32_t> eids;
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
                            allocator->Release(sid, nid);
                        }
                    }
                    // check if there are outputs that can be freeded immediately
                    // these output are not referenced by any operator.
                    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
                        uint32_t eid = idx.entry_id(nid, index);
                        auto sid = storage[eid];
                        if (sid >= 0 && storage_ref_count[sid] == 0) {
                            allocator->Release(sid, nid);
                            // use -2 to indicate that the node was never touched.
                            storage_inplace_index[eid] = -2;
                        }
                        if (storage[eid] == GraphAllocator::kBadStorageID) {
                            ++num_not_allocated;
                        }
                    }
                }
                return num_not_allocated;
            }

            /*
             * Internal method to perform the memory allocation for a graph
             * */
            size_t AllocMemory2(const Graph& ret, const IndexedGraph& idx,
                const std::pair<uint32_t, uint32_t>& node_range,
                StorageVector* storage_ptr,
                std::vector<int>* storage_inplace_index_ptr,
                const std::vector<uint32_t>& entry_ref_count,
                GraphAllocator* allocator,
                std::vector<size_t>* vsid_size_ptr,
                std::vector<size_t>* ventry_offset_ptr
                ) 
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
                std::vector<GraphAllocator::StorageID> storage_ref_count(idx.num_node_entries(), 0);

                std::vector<ChunkSlice> chunk_storage(idx.num_node_entries());
                std::unordered_map<ChunkPtr, int32_t> chunk_ref_count;
                PlanChunk(ret, idx, node_range, &chunk_storage,
                    storage_inplace_index_ptr, entry_ref_count, &chunk_ref_count);

                auto& vsid_size = *vsid_size_ptr;
                auto& ventry_offset = *ventry_offset_ptr;

                std::unordered_map<ChunkPtr, GraphAllocator::StorageID> chunk_allocated;
                for (uint32_t nid = node_range.first; nid < node_range.second; ++nid) {
                    const auto& inode = idx[nid];
                    if (inode.source->is_variable()) continue;

                    // normal allocation
                    const int dev_id = (device_vec != nullptr) ? device_vec->at(nid) : 0;
                    // sort output nodes based on size before allocating output
                    std::multimap<size_t, uint32_t> eids;
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
                        const ChunkSlice ck_entry = chunk_storage[eid].lower();
                        const ChunkPtr root_chunk = ck_entry.source();
                        auto sid = GraphAllocator::kBadStorageID;
                        auto it = chunk_allocated.find(root_chunk);
                        if (it == chunk_allocated.end() || it->second == GraphAllocator::kBadStorageID) {
                            sid = allocator->Request(dev_id, root_chunk->size(), nid);
                            chunk_allocated[root_chunk] = sid;
                            storage_ref_count[sid] = chunk_ref_count[root_chunk];
                            vsid_size[sid] = std::max(root_chunk->size(), vsid_size[sid]);

                            std::cout << "Alloc chunk=" << sid << "\ttotal=" << vsid_size[sid] 
                                << " \teid=" << eid << " \tsize=" 
                                << shape_vec[eid].Size() * mshadow::mshadow_sizeof(dtype_vec[eid]) << std::endl;
                        } 
                        else {
                            sid = it->second;

                            std::cout << "Share chunk=" << sid << "\ttotal=" << vsid_size[sid]
                                << " \teid=" << eid << " \tsize="
                                << shape_vec[eid].Size() * mshadow::mshadow_sizeof(dtype_vec[eid]) << std::endl;
                        }
                        storage[eid] = sid;
                        ventry_offset[eid] = ck_entry.offset();
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
                        if (sid < 0) continue;
                        // if we decrease it to zero, means we are ready to relase
                        --storage_ref_count[sid];
                        if (storage_ref_count[sid] == 0) {
                            allocator->Release(sid, nid);
                        }
                    }

                    // check if there are outputs that can be freeded immediately
                    // these output are not referenced by any operator.
                    for (uint32_t index = 0; index < inode.source->num_outputs(); ++index) {
                        uint32_t eid = idx.entry_id(nid, index);
                        auto sid = storage[eid];
                        if (sid >= 0 && storage_ref_count[sid] == 0) {
                            allocator->Release(sid, nid);
                            // use -2 to indicate that the node was never touched.
                            CHECK_EQ(storage_inplace_index[eid], -2); // qiuhan 注意这里！！ 该输出完全不会被用到
                        }
                        if (storage[eid] == GraphAllocator::kBadStorageID) {
                            ++num_not_allocated;
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
                std::pair<uint32_t, uint32_t> node_range = { 0, idx.num_nodes() };
                if (ret.attrs.count("node_range")) {
                    node_range = ret.MoveCopyAttr<std::pair<uint32_t, uint32_t> >("node_range");
                }

                // reference counter of each node
                std::vector<uint32_t> ref_count;
                // step 1: initialize reference count
                if (ret.attrs.count("ref_count") != 0) {
                    ref_count = ret.MoveCopyAttr<std::vector<uint32_t> >("ref_count");
                }
                else {
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

                    for (const auto& e : idx.outputs()) {
                        ++ref_count[idx.entry_id(e)];
                    }
                }

                // step 2: allocate memory.
                StorageVector storage;
                if (ret.attrs.count("storage") != 0) {
                    storage = ret.MoveCopyAttr<StorageVector>("storage");
                }
                else {
                    storage.resize(idx.num_node_entries(), -1);
                }

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

                    std::vector<size_t> ventry_offset(idx.num_node_entries(), 0);
                    std::vector<size_t> vsid_size(idx.num_node_entries(), 0);

                    // number of entries that are not statically allocated.
                    size_t storage_num_not_allocated =
                        AllocMemory(ret, idx, node_range, &storage_vec, &storage_inplace_index,
                            ref_count, &allocator);

                    size_t storage_allocated_bytes = allocator.TotalAllocBytes();
                    // Choose the plan which leads to minimal memory usage
                    if (min_allocated_bytes > storage_allocated_bytes) {
                        ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
                        ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
                        ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(storage_allocated_bytes);
                        ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(storage_num_not_allocated);
                        ret.attrs["sid_size"] = std::make_shared<any>(vsid_size);
                        ret.attrs["entry_offset"] = std::make_shared<any>(ventry_offset);
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
                .provide_graph_attr("storage_inplace_index")
                .provide_graph_attr("sid_size")
                .provide_graph_attr("entry_offset");



            Graph PlanMemoryFused(Graph ret) {
                // setup ref counter
                const IndexedGraph& idx = ret.indexed_graph();
                static auto& fignore_inputs = Op::GetAttr<FIgnoreInputs>("FIgnoreInputs");
                std::pair<uint32_t, uint32_t> node_range = { 0, idx.num_nodes() };
                if (ret.attrs.count("node_range")) {
                    node_range = ret.MoveCopyAttr<std::pair<uint32_t, uint32_t> >("node_range");
                }

                // reference counter of each node
                std::vector<uint32_t> ref_count;
                // step 1: initialize reference count
                if (ret.attrs.count("ref_count") != 0) {
                    ref_count = ret.MoveCopyAttr<std::vector<uint32_t> >("ref_count");
                }
                else {
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

                    for (const auto& e : idx.outputs()) {
                        ++ref_count[idx.entry_id(e)];
                    }
                }

                // step 2: allocate memory.
                StorageVector storage;
                if (ret.attrs.count("storage") != 0) {
                    storage = ret.MoveCopyAttr<StorageVector>("storage");
                }
                else {
                    storage.resize(idx.num_node_entries(), -1);
                }

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

                    std::vector<size_t> ventry_offset(idx.num_node_entries(), 0);
                    std::vector<size_t> vsid_size(idx.num_node_entries(), 0);

                    // number of entries that are not statically allocated.
                    size_t storage_num_not_allocated =
                        AllocMemory2(ret, idx, node_range, &storage_vec, &storage_inplace_index,
                            ref_count, &allocator, &vsid_size, &ventry_offset);

                    size_t storage_allocated_bytes = allocator.TotalAllocBytes();
                    // Choose the plan which leads to minimal memory usage
                    if (min_allocated_bytes > storage_allocated_bytes) {
                        ret.attrs["storage_id"] = std::make_shared<any>(std::move(storage_vec));
                        ret.attrs["storage_inplace_index"] = std::make_shared<any>(std::move(storage_inplace_index));
                        ret.attrs["storage_allocated_bytes"] = std::make_shared<any>(storage_allocated_bytes);
                        ret.attrs["storage_num_not_allocated"] = std::make_shared<any>(storage_num_not_allocated);
                        ret.attrs["sid_size"] = std::make_shared<any>(vsid_size);
                        ret.attrs["entry_offset"] = std::make_shared<any>(ventry_offset);
                        min_allocated_bytes = storage_allocated_bytes;
                    }

                    if (max_match_range == 0) {
                        break;
                    }
                }
                return ret;
            }

            NNVM_REGISTER_PASS(PlanMemoryFused)
                .describe("FORWARD ONLY ! Plan the memory allocation of each node entries.")
                .set_body(PlanMemoryFused)
                .set_change_graph(false)
                .depend_graph_attr("dtype")
                .depend_graph_attr("shape")
                .provide_graph_attr("storage_id")
                .provide_graph_attr("storage_inplace_index")
                .provide_graph_attr("sid_size")
                .provide_graph_attr("entry_offset");

        }  // namespace
    }  // namespace pass
}  // namespace nnvm
