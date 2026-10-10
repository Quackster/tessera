#include "core/think_budget.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>

namespace tessera::core {

namespace {

// True when the recent ids end with the tag sequence.
bool EndsWith(const std::deque<std::uint32_t>& recent,
              std::span<const std::uint32_t> tag) {
  if (recent.size() < tag.size()) {
    return false;
  }
  return std::equal(tag.begin(), tag.end(),
                    recent.end() - static_cast<std::ptrdiff_t>(tag.size()));
}

}  // namespace

void ThinkBudget::engage(std::span<const std::uint32_t> open_ids,
                         std::span<const std::uint32_t> close_ids,
                         std::size_t budget) {
  open_.assign(open_ids.begin(), open_ids.end());
  close_.assign(close_ids.begin(), close_ids.end());
  budget_ = budget;
  count_ = 0;
  in_think_ = false;
  recent_.clear();
  window_ = std::max(open_.size(), close_.size());
}

bool ThinkBudget::engaged() const {
  return budget_ > 0 && !open_.empty() && !close_.empty();
}

void ThinkBudget::noteEmitted(std::uint32_t id) {
  if (!engaged()) {
    return;
  }
  recent_.push_back(id);
  while (recent_.size() > window_) {
    recent_.pop_front();
  }
  // The tag-completing token is markup, not thinking content: an open
  // match enters with a fresh count, a close match leaves without
  // counting, and only the tokens between them count.
  if (!in_think_) {
    if (EndsWith(recent_, open_)) {
      in_think_ = true;
      count_ = 0;
    }
    return;
  }
  if (EndsWith(recent_, close_)) {
    in_think_ = false;
    return;
  }
  ++count_;
}

bool ThinkBudget::closeOwed() const {
  return engaged() && in_think_ && count_ >= budget_;
}

std::size_t ThinkBudget::thinkCount() const {
  return count_;
}

bool ThinkBudget::inThink() const {
  return in_think_;
}

}  // namespace tessera::core
