#pragma once

#include <memory>
#include <vector>


namespace nnvm {
    namespace pass {

        class Chunk;
        using ChunkPtr = std::shared_ptr<Chunk>;

#define DISABLE_COPY_AND_ASSIGN(NAME_) \
                 private: \
                     NAME_(const NAME_ &); \
                     NAME_(const NAME_ &&); \
                     NAME_ & operator=(const NAME_ &); \
                     NAME_ & operator=(const NAME_ &&);

        class Chunk
            : public std::enable_shared_from_this<Chunk>
        {
            DISABLE_COPY_AND_ASSIGN(Chunk)

        public:
            class ChunkSlice
            {
            public:
                ChunkSlice(const ChunkPtr& chunk,
                    const size_t offset,
                    const size_t size)
                    : chunk_(chunk), offset_(offset), size_(size)
                {
                }

                ChunkSlice() { }

                ChunkSlice slice(const size_t offset, const size_t size) {
                    CHECK_LE(offset + offset_ + size, chunk_->size());
                    return ChunkSlice(chunk_, offset_ + offset, size);
                }

                ChunkPtr source() const {
                    return chunk_;
                }

                bool is_null() const {
                    return chunk_ == nullptr;
                }

                size_t offset() const {
                    return offset_;
                }

                size_t size() const {
                    return size_;
                }

                ChunkSlice lower() const 
                {
                    if (chunk_) {
                        return ChunkSlice(chunk_->root_parent(), offset_ + chunk_->root_offset(), size_);
                    }
                    else {
                        return ChunkSlice();
                    }
                }

                friend bool operator< (const ChunkSlice& lhs, const ChunkSlice& rhs)
                {
                    if (lhs.source() < rhs.source()) {
                        return true;
                    }
                    else if (lhs.source() > rhs.source()) {
                        return false;
                    }

                    if (lhs.size() < rhs.size()) {
                        return true;
                    }
                    else if (lhs.size() > rhs.size()) {
                        return false;
                    }

                    if (lhs.offset() < rhs.offset()) {
                        return true;
                    }

                    return false;
                }

            private:
                ChunkPtr chunk_;
                size_t offset_{ 0 };
                size_t size_{ 0 };
            };

            enum class ExpandDirection {
                kExpandHead,
                kExpandTail
            };

            static ChunkPtr make(const size_t size) {
                return ChunkPtr(new Chunk(size));
            }

            size_t size() const {
                return size_;
            }

            ChunkSlice slice(const size_t offset, const size_t size) {
                CHECK_LE(offset + size, size_);
                return ChunkSlice(shared_from_this(), offset, size);
            }

            ChunkPtr expand(const size_t new_size, const ExpandDirection direct)
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

            ChunkPtr embed_in(const ChunkPtr& parent, const size_t offset)
            {
                CHECK(!parent_);
                CHECK_LE(offset_ + size_, parent->size());

                parent_ = parent;
                offset_ = offset;

                return parent_;
            }

            ChunkPtr parent() {
                return parent_;
            }

            ChunkPtr root_parent()
            {
                ChunkPtr gp = shared_from_this();
                while (gp->parent()) {
                    gp = gp->parent();
                }
                return gp;
            }

            size_t offset() const {
                return offset_;
            }

            size_t root_offset() const
            {
                size_t offset = offset_;
                ChunkPtr gp = parent_;
                // 这里不用递归防止爆栈
                while (gp) {
                    offset += gp->offset();
                    gp = gp->parent();
                }
                return offset;
            }

        private:
            Chunk(const size_t size)
                : size_(size)
            {
            }

            size_t size_{ 0 };
            size_t offset_{ 0 };
            ChunkPtr parent_;
        };

        using ChunkSlice = Chunk::ChunkSlice;
    }
}