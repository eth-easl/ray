// Copyright 2017 The Ray Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ray/object_manager/object_buffer_pool.h"

#include "ray/common/status.h"
#include "ray/util/logging.h"

namespace ray {

ObjectBufferPool::ObjectBufferPool(const std::string &store_socket_name,
                                   uint64_t chunk_size)
    : default_chunk_size_(chunk_size) {
  store_socket_name_ = store_socket_name;
  RAY_CHECK_OK(store_client_.Connect(store_socket_name_.c_str(), "", 0, 300));
}

ObjectBufferPool::~ObjectBufferPool() {
  // Abort everything in progress.
  auto get_buf_state_copy = get_buffer_state_;
  for (const auto &pair : get_buf_state_copy) {
    AbortGet(pair.first);
  }
  auto create_buf_state_copy = create_buffer_state_;
  for (const auto &pair : create_buf_state_copy) {
    AbortCreate(pair.first);
  }
  RAY_CHECK(get_buffer_state_.empty());
  RAY_CHECK(create_buffer_state_.empty());
  RAY_CHECK_OK(store_client_.Disconnect());
}

uint64_t ObjectBufferPool::GetNumChunks(uint64_t data_size) {
  return (data_size + default_chunk_size_ - 1) / default_chunk_size_;
}

uint64_t ObjectBufferPool::GetBufferLength(uint64_t chunk_index, uint64_t data_size) {
  return (chunk_index + 1) * default_chunk_size_ > data_size
             ? data_size % default_chunk_size_
             : default_chunk_size_;
}

std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status> ObjectBufferPool::GetChunk(
    const ObjectID &object_id, uint64_t data_size, uint64_t metadata_size,
    uint64_t chunk_index, std::unordered_map<ObjectID, double> &read_objects_total) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  if (get_buffer_state_.count(object_id) == 0) {

    //RAY_LOG(INFO) << "object: " << object_id << "not in get_buffer_state_";
    plasma::ObjectBuffer object_buffer;

    double start_time = absl::GetCurrentTimeNanos() / 1e9;

    // TODO[fot]: measure time here! for all chunks of the object!
    RAY_CHECK_OK(
        store_client_.Get(&object_id, 1, 0, &object_buffer, /*is_from_worker=*/false));
    double end_time = absl::GetCurrentTimeNanos() / 1e9;
    if (read_objects_total.find(object_id) == read_objects_total.end()) {
      read_objects_total[object_id] = end_time - start_time;
    }
    else {
       read_objects_total[object_id] += end_time - start_time;
    }

    if (object_buffer.data == nullptr) {
      RAY_LOG(INFO)
          << "Failed to get a chunk of the object: " << object_id
          << ". It is mostly because the object is already evicted or spilled when the "
             "pull request is received. The caller will retry the pull request again.";
      return std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status>(
          errored_chunk_,
          ray::Status::IOError("Unable to obtain object chunk, object not local."));
    }
    RAY_CHECK(object_buffer.metadata->Data() ==
              object_buffer.data->Data() + object_buffer.data->Size());
    RAY_CHECK(data_size == static_cast<uint64_t>(object_buffer.data->Size() +
                                                 object_buffer.metadata->Size()));
    auto *data = object_buffer.data->Data();
    uint64_t num_chunks = GetNumChunks(data_size);
    get_buffer_state_.emplace(
        std::piecewise_construct, std::forward_as_tuple(object_id),
        std::forward_as_tuple(BuildChunks(object_id, data, data_size)));
    RAY_CHECK(get_buffer_state_[object_id].chunk_info.size() == num_chunks);
  }
  get_buffer_state_[object_id].references++;
  return std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status>(
      get_buffer_state_[object_id].chunk_info[chunk_index], ray::Status::OK());
}

void ObjectBufferPool::ReleaseGetChunk(const ObjectID &object_id, uint64_t chunk_index) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  GetBufferState &buffer_state = get_buffer_state_[object_id];
  buffer_state.references--;
  if (buffer_state.references == 0) {
    RAY_CHECK_OK(store_client_.Release(object_id));
    get_buffer_state_.erase(object_id);
  }
}

void ObjectBufferPool::AbortGet(const ObjectID &object_id) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  RAY_CHECK_OK(store_client_.Release(object_id));
  get_buffer_state_.erase(object_id);
}

std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status> ObjectBufferPool::CreateChunk(
    const ObjectID &object_id, const rpc::Address &owner_address, uint64_t data_size,
    uint64_t metadata_size, uint64_t chunk_index) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  if (create_buffer_state_.count(object_id) == 0) {
    int64_t object_size = data_size - metadata_size;
    // Try to create shared buffer.
    std::shared_ptr<Buffer> data;
    Status s = store_client_.TryCreateImmediately(object_id, owner_address, object_size,
                                                  NULL, metadata_size, &data);
    std::vector<boost::asio::mutable_buffer> buffer;
    if (!s.ok()) {
      // Create failed. The object may already exist locally. If something else went
      // wrong, another chunk will succeed in creating the buffer, and this
      // chunk will eventually make it here via pull requests.
      return std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status>(
          errored_chunk_, ray::Status::IOError(s.message()));
    }
    // Read object into store.
    uint8_t *mutable_data = data->Data();
    uint64_t num_chunks = GetNumChunks(data_size);
    create_buffer_state_.emplace(
        std::piecewise_construct, std::forward_as_tuple(object_id),
        std::forward_as_tuple(BuildChunks(object_id, mutable_data, data_size)));
    RAY_LOG(INFO) << "Created object " << object_id
                   << " in plasma store, number of chunks: " << num_chunks
                   << ", chunk index: " << chunk_index;
    RAY_CHECK(create_buffer_state_[object_id].chunk_info.size() == num_chunks);
  }
  if (create_buffer_state_[object_id].chunk_state[chunk_index] !=
      CreateChunkState::AVAILABLE) {
    // There can be only one reference to this chunk at any given time.
    return std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status>(
        errored_chunk_,
        ray::Status::IOError("Chunk already received by a different thread."));
  }
  create_buffer_state_[object_id].chunk_state[chunk_index] = CreateChunkState::REFERENCED;
  return std::pair<const ObjectBufferPool::ChunkInfo &, ray::Status>(
      create_buffer_state_[object_id].chunk_info[chunk_index], ray::Status::OK());
}

void ObjectBufferPool::AbortCreateChunk(const ObjectID &object_id,
                                        const uint64_t chunk_index) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  RAY_CHECK(create_buffer_state_[object_id].chunk_state[chunk_index] ==
            CreateChunkState::REFERENCED);
  create_buffer_state_[object_id].chunk_state[chunk_index] = CreateChunkState::AVAILABLE;
  if (create_buffer_state_[object_id].num_seals_remaining ==
      create_buffer_state_[object_id].chunk_state.size()) {
    // If chunk_state is AVAILABLE at every chunk_index and
    // num_seals_remaining == num_chunks, this is back to the initial state
    // right before the first CreateChunk.
    bool abort = true;
    for (auto chunk_state : create_buffer_state_[object_id].chunk_state) {
      abort &= chunk_state == CreateChunkState::AVAILABLE;
    }
    if (abort) {
      AbortCreate(object_id);
    }
  }
}

void ObjectBufferPool::SealChunk(const ObjectID &object_id, const uint64_t chunk_index, std::unordered_map<ObjectID, double>& pull_objects_start, std::unordered_map<ObjectID, double>& write_objects_total) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  auto it = create_buffer_state_.find(object_id);
  if (it == create_buffer_state_.end() ||
      it->second.chunk_state[chunk_index] != CreateChunkState::REFERENCED) {
    RAY_LOG(DEBUG) << "Object " << object_id << " aborted due to OOM before chunk "
                   << chunk_index << " could be sealed";
    return;
  }
  it->second.chunk_state[chunk_index] = CreateChunkState::SEALED;
  it->second.num_seals_remaining--;
  if (it->second.num_seals_remaining == 0) {

    RAY_CHECK_OK(store_client_.Seal(object_id));
    RAY_CHECK_OK(store_client_.Release(object_id));
    create_buffer_state_.erase(it);
    RAY_LOG(INFO) << "Have received all chunks for object " << object_id
                   << ", last chunk index: " << chunk_index;

    if (pull_objects_start.find(object_id) != pull_objects_start.end()) {

      double end_time =  absl::GetCurrentTimeNanos() / 1e9;
      RAY_LOG(DEBUG) << "Getting the object: " << object_id << " took " <<  end_time - pull_objects_start[object_id] << "sec";
      RAY_LOG(DEBUG) << "Writing the object to Plasma: " << object_id << " took " << write_objects_total[object_id] << "sec";
      pull_objects_start.erase(object_id);
      write_objects_total.erase(object_id);
    }

  }


}

void ObjectBufferPool::AbortCreate(const ObjectID &object_id) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  auto it = create_buffer_state_.find(object_id);
  if (it != create_buffer_state_.end()) {
    RAY_LOG(INFO) << "Not enough memory to create requested object " << object_id
                  << ", aborting";
    RAY_CHECK_OK(store_client_.Release(object_id));
    RAY_CHECK_OK(store_client_.Abort(object_id));
    create_buffer_state_.erase(object_id);
  }
}

std::vector<ObjectBufferPool::ChunkInfo> ObjectBufferPool::BuildChunks(
    const ObjectID &object_id, uint8_t *data, uint64_t data_size) {
  uint64_t space_remaining = data_size;
  std::vector<ChunkInfo> chunks;
  int64_t position = 0;
  while (space_remaining) {
    position = data_size - space_remaining;
    if (space_remaining < default_chunk_size_) {
      chunks.emplace_back(chunks.size(), data + position, space_remaining);
      space_remaining = 0;
    } else {
      chunks.emplace_back(chunks.size(), data + position, default_chunk_size_);
      space_remaining -= default_chunk_size_;
    }
  }
  return chunks;
}

void ObjectBufferPool::FreeObjects(const std::vector<ObjectID> &object_ids) {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  RAY_CHECK_OK(store_client_.Delete(object_ids));
}

std::string ObjectBufferPool::DebugString() const {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  std::stringstream result;
  result << "BufferPool:";
  result << "\n- get buffer state map size: " << get_buffer_state_.size();
  result << "\n- create buffer state map size: " << create_buffer_state_.size();
  return result.str();
}

}  // namespace ray
