#include "tessera/mtp_fetch.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <system_error>
#include <vector>

#include "core/files.hpp"
#include "core/http.hpp"
#include "core/json.hpp"
#include "core/sha256.hpp"

namespace tessera {

namespace {

// The pinned checkpoint commit (its `sha` from
// https://huggingface.co/api/models/Qwen/Qwen3.8-Flash-Next), so every
// install reads the same tensors; TESSERA_MTP_REVISION overrides it.
constexpr std::string_view kPinnedRevision =
    "de4b8e4d43b917e7706784d8bb445c9af86a3540";
constexpr std::string_view kDefaultEndpoint = "https://huggingface.co";
constexpr std::string_view kRepoPath = "/Qwen/Qwen3.8-Flash-Next/resolve/";
// A shard header read in one request, and the tensor download chunk. Both
// bound a single buffer, never the whole shard or tensor.
constexpr std::uint64_t kShardHeaderChunk = 64u << 20;
constexpr std::string_view kIndexName = "model.safetensors.index.json";

std::string EnvOrDefault(const char* name, std::string_view fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || *value == '\0') {
    return std::string(fallback);
  }
  return value;
}

std::string Endpoint() {
  std::string endpoint = EnvOrDefault("HF_ENDPOINT", kDefaultEndpoint);
  while (!endpoint.empty() && endpoint.back() == '/') {
    endpoint.pop_back();
  }
  return endpoint;
}

std::string Revision() {
  return EnvOrDefault("TESSERA_MTP_REVISION", kPinnedRevision);
}

std::string Repo() {
  return Endpoint() + std::string(kRepoPath) + Revision() + "/";
}

std::string PinnedRepo() {
  return Endpoint() + std::string(kRepoPath) + std::string(kPinnedRevision) +
         "/";
}

// The digests apply only when this run reads the pinned revision.
bool Pinned() { return Repo() == PinnedRepo(); }

// Parse a JSON value's unsigned integer. JSON numbers here are byte
// offsets and element counts, all below 2^53, so a double carries them.
std::uint64_t JsonUint(const core::Json* value) {
  return value == nullptr ? 0 : static_cast<std::uint64_t>(value->AsNumber());
}

// One MTP tensor's location in the checkpoint: its shard, byte range and
// shape/dtype.
struct InventoryRow {
  std::string name;
  std::string shard;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  std::uint64_t bytes = 0;
};

struct Inventory {
  std::string repo;
  std::uint64_t total_bytes = 0;
  std::vector<InventoryRow> tensors;
};

const MtpTensorDigest* FindDigest(std::string_view name) {
  for (const MtpTensorDigest& digest : MtpTensorDigests()) {
    if (digest.name == name) {
      return &digest;
    }
  }
  return nullptr;
}

// A single HTTP range read that must answer 206 with the exact
// Content-Range, then the body. Any other status or header is a mirror
// that ignored the Range header, so the bytes are not what was asked for.
std::expected<std::vector<std::byte>, StatusCode> GetRange(
    const std::string& url, std::uint64_t first, std::uint64_t last) {
  auto reply = core::HttpGet(core::HttpRequest{url, core::ByteRange{first, last}});
  if (!reply) {
    return std::unexpected(reply.error());
  }
  if (reply->status != 206) {
    return std::unexpected(StatusCode::NetworkError);
  }
  char want[64];
  std::snprintf(want, sizeof(want), "bytes %llu-%llu/",
                static_cast<unsigned long long>(first),
                static_cast<unsigned long long>(last));
  if (reply->content_range.rfind(want, 0) != 0) {
    return std::unexpected(StatusCode::NetworkError);
  }
  if (reply->body.size() != last - first + 1) {
    return std::unexpected(StatusCode::NetworkError);
  }
  return reply->body;
}

std::expected<core::Json, StatusCode> GetJson(const std::string& url) {
  auto reply = core::HttpGet(core::HttpRequest{url, std::nullopt});
  if (!reply) {
    return std::unexpected(reply.error());
  }
  if (reply->status != 200) {
    return std::unexpected(StatusCode::NetworkError);
  }
  const std::string text(reinterpret_cast<const char*>(reply->body.data()),
                         reply->body.size());
  auto parsed = core::Json::Parse(text);
  if (parsed == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return std::move(*parsed);
}

// Build the inventory of every `mtp.*` tensor: fetch the shard index, then
// each MTP shard's header, and record the byte range of every MTP tensor.
std::expected<Inventory, StatusCode> BuildInventory() {
  auto index = GetJson(Repo() + std::string(kIndexName));
  if (!index) {
    return std::unexpected(index.error());
  }
  const core::Json* weight_map = index->Find("weight_map");
  if (weight_map == nullptr || !weight_map->isObject()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::map<std::string, std::string> mtp_to_shard;
  for (const auto& [name, shard] : weight_map->AsObject()) {
    if (shard.isString() && name.rfind("mtp.", 0) == 0) {
      mtp_to_shard.emplace(name, shard.AsString());
    }
  }
  std::map<std::string, std::vector<std::string>> by_shard;
  for (const auto& [name, shard] : mtp_to_shard) {
    by_shard[shard].push_back(name);
  }
  Inventory inventory;
  inventory.repo = Repo();
  for (const auto& [shard, names] : by_shard) {
    const std::string shard_url = Repo() + shard;
    auto length_bytes = GetRange(shard_url, 0, 7);
    if (!length_bytes) {
      return std::unexpected(length_bytes.error());
    }
    std::uint64_t header_len = 0;
    std::memcpy(&header_len, length_bytes->data(), 8);
    if (header_len == 0 || header_len > kShardHeaderChunk) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    auto header_bytes =
        GetRange(shard_url, 8, 8 + header_len - 1);
    if (!header_bytes) {
      return std::unexpected(header_bytes.error());
    }
    const std::string header_text(
        reinterpret_cast<const char*>(header_bytes->data()), header_bytes->size());
    auto header = core::Json::Parse(header_text);
    if (header == nullptr || !header->isObject()) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    const std::uint64_t data_start = 8 + header_len;
    for (const std::string& name : names) {
      const core::Json* meta = header->Find(name);
      if (meta == nullptr || !meta->isObject()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      const core::Json* offsets = meta->Find("data_offsets");
      const core::Json* shape = meta->Find("shape");
      const core::Json* dtype = meta->Find("dtype");
      if (offsets == nullptr || !offsets->isArray() ||
          offsets->AsArray().size() != 2 || shape == nullptr ||
          !shape->isArray() || dtype == nullptr || !dtype->isString()) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      InventoryRow row;
      row.name = name;
      row.shard = shard;
      row.dtype = dtype->AsString();
      for (const core::Json& dim : shape->AsArray()) {
        row.shape.push_back(JsonUint(&dim));
      }
      const std::uint64_t begin = JsonUint(&offsets->AsArray()[0]);
      const std::uint64_t stop = JsonUint(&offsets->AsArray()[1]);
      if (stop <= begin) {
        return std::unexpected(StatusCode::MalformedFile);
      }
      row.start = data_start + begin;
      row.end = data_start + stop - 1;
      row.bytes = stop - begin;
      inventory.total_bytes += row.bytes;
      inventory.tensors.push_back(std::move(row));
    }
  }
  if (inventory.tensors.size() != MtpTensorDigests().size()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::sort(inventory.tensors.begin(), inventory.tensors.end(),
            [](const InventoryRow& a, const InventoryRow& b) {
              return a.name < b.name;
            });
  return inventory;
}

core::Json InventoryToJson(const Inventory& inventory) {
  core::Json root = core::Json::Object();
  root.Set("repo", core::Json::String(inventory.repo));
  root.Set("total_bytes",
           core::Json::Number(static_cast<double>(inventory.total_bytes)));
  core::Json list = core::Json::Array();
  for (const InventoryRow& row : inventory.tensors) {
    core::Json item = core::Json::Object();
    item.Set("name", core::Json::String(row.name));
    item.Set("shard", core::Json::String(row.shard));
    item.Set("dtype", core::Json::String(row.dtype));
    core::Json shape = core::Json::Array();
    for (const std::uint64_t dim : row.shape) {
      shape.Push(core::Json::Number(static_cast<double>(dim)));
    }
    item.Set("shape", std::move(shape));
    item.Set("start", core::Json::Number(static_cast<double>(row.start)));
    item.Set("end", core::Json::Number(static_cast<double>(row.end)));
    item.Set("bytes", core::Json::Number(static_cast<double>(row.bytes)));
    list.Push(std::move(item));
  }
  root.Set("tensors", std::move(list));
  return root;
}

std::expected<Inventory, StatusCode> InventoryFromJson(const core::Json& root) {
  Inventory inventory;
  const core::Json* repo = root.Find("repo");
  const core::Json* tensors = root.Find("tensors");
  if (repo == nullptr || !repo->isString() || tensors == nullptr ||
      !tensors->isArray()) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  inventory.repo = repo->AsString();
  inventory.total_bytes = JsonUint(root.Find("total_bytes"));
  for (const core::Json& item : tensors->AsArray()) {
    InventoryRow row;
    const core::Json* name = item.Find("name");
    const core::Json* shard = item.Find("shard");
    const core::Json* dtype = item.Find("dtype");
    const core::Json* shape = item.Find("shape");
    if (name == nullptr || !name->isString() || shard == nullptr ||
        !shard->isString() || dtype == nullptr || !dtype->isString() ||
        shape == nullptr || !shape->isArray()) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    row.name = name->AsString();
    row.shard = shard->AsString();
    row.dtype = dtype->AsString();
    for (const core::Json& dim : shape->AsArray()) {
      row.shape.push_back(JsonUint(&dim));
    }
    row.start = JsonUint(item.Find("start"));
    row.end = JsonUint(item.Find("end"));
    row.bytes = JsonUint(item.Find("bytes"));
    inventory.tensors.push_back(std::move(row));
  }
  return inventory;
}

std::filesystem::path InventoryPath(const std::filesystem::path& out) {
  return out / "mtp-inventory.json";
}

std::filesystem::path ManifestPath(const std::filesystem::path& out) {
  return out / "mtp-manifest.json";
}

std::filesystem::path TensorsDir(const std::filesystem::path& out) {
  return out / "tensors";
}

std::filesystem::path TensorPath(const std::filesystem::path& out,
                                 const std::string& name) {
  return TensorsDir(out) / (name + ".bin");
}

std::uint64_t FileSize(const std::filesystem::path& path) {
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  return ec ? 0 : size;
}

// Append one tensor's bytes, resuming from the on-disk size. The download
// is a sequence of bounded range reads, so a tensor larger than the
// available memory still completes. Each chunk is flushed, so an interrupted
// run resumes from exactly the bytes on disk.
std::expected<void, StatusCode> DownloadTensor(const MtpFetchOptions& options,
                                               const InventoryRow& row) {
  const std::filesystem::path path = TensorPath(options.out, row.name);
  const std::string url = Repo() + row.shard;
  std::uint64_t have = FileSize(path);
  if (have > row.bytes) {
    std::filesystem::remove(path);
    have = 0;
  }
  if (options.progress) {
    options.progress(row.name, have, row.bytes);
  }
  std::ofstream file(path, std::ios::binary | std::ios::app);
  if (!file) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::uint64_t pos = row.start + have;
  while (pos <= row.end) {
    const std::uint64_t stop = std::min(pos + kShardHeaderChunk - 1, row.end);
    auto chunk = GetRange(url, pos, stop);
    if (!chunk) {
      return std::unexpected(chunk.error());
    }
    file.write(reinterpret_cast<const char*>(chunk->data()),
               static_cast<std::streamsize>(chunk->size()));
    file.flush();
    if (!file) {
      return std::unexpected(StatusCode::MalformedFile);
    }
    pos = stop + 1;
    if (options.progress) {
      options.progress(row.name, pos - row.start, row.bytes);
    }
  }
  file.close();
  if (FileSize(path) != row.bytes) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  return {};
}

}  // namespace

std::span<const MtpTensorDigest> MtpTensorDigests() {
  static const std::array<MtpTensorDigest, 31> kDigests = {{
      {"mtp.fc_embedding.weight",
       "241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751"},
      {"mtp.fc_hidden.weight",
       "97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96"},
      {"mtp.hyper_connection_mixer.hc_norm.weight",
       "ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64"},
      {"mtp.hyper_connection_mixer.input_mix_weight_down.weight",
       "c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947"},
      {"mtp.hyper_connection_mixer.input_mix_weight_up.weight",
       "6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13"},
      {"mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
       "c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c"},
      {"mtp.layers.0.attn_hyper_connection.hc_norm.weight",
       "1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809"},
      {"mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
       "e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5"},
      {"mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight",
       "f80f1fd44e1183e7764ea496b11ff89ead9dd758f861d4e79a770bf54956b1a5"},
      {"mtp.layers.0.mlp.experts.down_proj",
       "539c46234f73bac70450f731f40e715f77ac32faf77def0f3f800e58aaabbbc0"},
      {"mtp.layers.0.mlp.experts.gate_up_proj",
       "839e98f9642ebc884b178abd2090b6dc8b0bcd81ed92d7f7a08314c6a68f96d5"},
      {"mtp.layers.0.mlp.gate.weight",
       "f0d35fa41b360a08bc0c5b1e5766aa42df960291716ee189208b5d2599462c20"},
      {"mtp.layers.0.mlp.shared_expert.down_proj.weight",
       "d2ced8c81367cc494599d1a8cdc279c3dd6579b628ded05acd7495e99cee01d8"},
      {"mtp.layers.0.mlp.shared_expert.gate_proj.weight",
       "f4a81109165853b3b64f36d809f98094b599763364070f5fa079200e1c386d4c"},
      {"mtp.layers.0.mlp.shared_expert.up_proj.weight",
       "172586861eea53efbff10afc43ded5b6e49c83879e9efc01e089c652bbb0a156"},
      {"mtp.layers.0.mlp.shared_expert_gate.weight",
       "acdd306d9996d356e4667a16ff63af6fef086fc7f570f1bc00bbe882a2a93f6e"},
      {"mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
       "c5782dd92b479a1f8ef43d683fe9c917fd9e8fba81f23c1aae8817bfe4b476a6"},
      {"mtp.layers.0.mlp_hyper_connection.hc_norm.weight",
       "21a504876b5e7b788301557c68dfb757999430eba36c66ec8ab96a63604dbfd1"},
      {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
       "b71240036dea954e034d5ae04e097bcd7ac51cb28a44e4c93d198af3447278a1"},
      {"mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
       "40756866bf2a5e82c28e407f717b6fa07614d593f78e79c7ceb2cd4ffe6dda94"},
      {"mtp.layers.0.self_attn.indexer.index_qk_proj.weight",
       "2befafae6e944c1fc651fd73d4d527eb6f138a6c1679065b003128ee3150e859"},
      {"mtp.layers.0.self_attn.indexer.k_layernorm.weight",
       "2db621f6a08b4085756fc69d60a4a1bd1d7fbf5d406456d2477075842efaae54"},
      {"mtp.layers.0.self_attn.indexer.q_layernorm.weight",
       "afaf59f68e251939970ffdeacad0667dffc657a522bea62cdc44a52816fc49d6"},
      {"mtp.layers.0.self_attn.k_norm.weight",
       "8f924b0357cf936e14fc34518205d000e1423aa60626827df81b0804fae8cdfd"},
      {"mtp.layers.0.self_attn.k_proj.weight",
       "1a295dc2d6e633e808aeb19e00c97856a1817bbfcc83f5b610e2bfcacfcb8c38"},
      {"mtp.layers.0.self_attn.o_proj.weight",
       "9d2fcf0ac6ac19de87f638f710cfd1af6ad14141e75b7d3d5d29fbd06f374219"},
      {"mtp.layers.0.self_attn.q_norm.weight",
       "57957aa1e7c5a3ac628bff0ec5522ad55a39b8c0b8bd2d324ae640e3771617ef"},
      {"mtp.layers.0.self_attn.q_proj.weight",
       "5220aef3d288f40759f7356af9ef453e5ac0256a38749a9bb803a4e9db58150d"},
      {"mtp.layers.0.self_attn.v_proj.weight",
       "39cb12f5424e965fa71fb81db0ee230c701b7b1e0b72c36641fde7e0e41bff41"},
      {"mtp.pre_fc_norm_embedding.weight",
       "04c4a570850e06f2d8913da8220d54d4c7f87db6eb6d45480b938e8ba41d6a86"},
      {"mtp.pre_fc_norm_hidden.weight",
       "b933b44559879449b1baf2af8385c5c237347307e3fd452394870cb53e6eab6d"},
  }};
  return kDigests;
}

std::expected<void, StatusCode> MtpInventory(const MtpFetchOptions& options) {
  if (!core::HttpAvailable()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  std::error_code ec;
  std::filesystem::create_directories(options.out, ec);
  if (ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto inventory = BuildInventory();
  if (!inventory) {
    return std::unexpected(inventory.error());
  }
  const std::string text = InventoryToJson(*inventory).Dump();
  const auto* bytes = reinterpret_cast<const std::byte*>(text.data());
  return core::WriteFile(InventoryPath(options.out),
                         std::span<const std::byte>(bytes, text.size()));
}

std::expected<void, StatusCode> MtpFetch(const MtpFetchOptions& options) {
  if (!core::HttpAvailable()) {
    return std::unexpected(StatusCode::UnsupportedFeature);
  }
  std::error_code ec;
  std::filesystem::create_directories(options.out, ec);
  if (ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  std::filesystem::create_directories(TensorsDir(options.out), ec);
  if (ec) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  // An inventory from another repository or revision has the wrong byte
  // ranges, so read it again before fetching.
  bool same = false;
  {
    auto text = core::ReadFile(InventoryPath(options.out));
    if (text) {
      const std::string json(reinterpret_cast<const char*>(text->data()),
                             text->size());
      auto parsed = core::Json::Parse(json);
      if (parsed != nullptr) {
        auto loaded = InventoryFromJson(*parsed);
        same = loaded && loaded->repo == Repo();
      }
    }
  }
  if (!same) {
    auto built = MtpInventory(options);
    if (!built) {
      return std::unexpected(built.error());
    }
  }
  auto text = core::ReadFile(InventoryPath(options.out));
  if (!text) {
    return std::unexpected(text.error());
  }
  const std::string json(reinterpret_cast<const char*>(text->data()),
                         text->size());
  auto parsed = core::Json::Parse(json);
  if (parsed == nullptr) {
    return std::unexpected(StatusCode::MalformedFile);
  }
  auto inventory = InventoryFromJson(*parsed);
  if (!inventory) {
    return std::unexpected(inventory.error());
  }
  core::Json manifest = core::Json::Array();
  for (const InventoryRow& row : inventory->tensors) {
    if (!options.only.empty() && row.name.find(options.only) == std::string::npos) {
      continue;
    }
    const MtpTensorDigest* digest = FindDigest(row.name);
    const std::string want =
        (Pinned() && digest != nullptr) ? std::string(digest->sha256) : "";
    const std::filesystem::path path = TensorPath(options.out, row.name);
    // A kept tensor from another revision cannot be resumed: it is removed
    // and fetched whole.
    if (!same && FileSize(path) != 0) {
      if (want.empty() || FileSize(path) != row.bytes) {
        std::filesystem::remove(path);
      } else {
        auto existing = core::Sha256File(path);
        if (!existing || *existing != want) {
          std::filesystem::remove(path);
        }
      }
    }
    std::string got;
    for (int attempt = 0; attempt < 2; ++attempt) {
      auto downloaded = DownloadTensor(options, row);
      if (!downloaded) {
        return std::unexpected(downloaded.error());
      }
      auto computed = core::Sha256File(path);
      if (!computed) {
        return std::unexpected(computed.error());
      }
      got = *computed;
      if (want.empty() || got == want) {
        break;
      }
      if (attempt == 1) {
        std::filesystem::remove(path);
        return std::unexpected(StatusCode::MalformedFile);
      }
      std::filesystem::remove(path);
    }
    core::Json item = core::Json::Object();
    item.Set("name", core::Json::String(row.name));
    item.Set("shard", core::Json::String(row.shard));
    item.Set("dtype", core::Json::String(row.dtype));
    core::Json shape = core::Json::Array();
    for (const std::uint64_t dim : row.shape) {
      shape.Push(core::Json::Number(static_cast<double>(dim)));
    }
    item.Set("shape", std::move(shape));
    item.Set("bytes", core::Json::Number(static_cast<double>(row.bytes)));
    item.Set("sha256", core::Json::String(got));
    manifest.Push(std::move(item));
  }
  const std::string out_text = manifest.Dump();
  const auto* out_bytes = reinterpret_cast<const std::byte*>(out_text.data());
  return core::WriteFile(ManifestPath(options.out),
                         std::span<const std::byte>(out_bytes, out_text.size()));
}

std::expected<std::vector<std::string>, StatusCode> MtpVerify(
    const std::filesystem::path& out) {
  std::vector<std::string> bad;
  if (!Pinned() || !std::filesystem::is_directory(TensorsDir(out))) {
    return bad;
  }
  for (const MtpTensorDigest& digest : MtpTensorDigests()) {
    const std::filesystem::path path = TensorPath(out, std::string(digest.name));
    if (!std::filesystem::exists(path)) {
      bad.emplace_back(digest.name);
      continue;
    }
    auto computed = core::Sha256File(path);
    if (!computed || *computed != digest.sha256) {
      bad.emplace_back(digest.name);
    }
  }
  return bad;
}

}  // namespace tessera
