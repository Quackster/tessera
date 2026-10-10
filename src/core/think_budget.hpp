#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <span>
#include <vector>

// Thinking-protocol markers: the chat template opens the think block
// with "<think>" (often trailing the prompt) and the model closes it
// with "</think>". Format constants, never configurable. Shared with
// the serving layer, which splits the same spans from reply text.
namespace tessera::core {

inline constexpr char kThinkOpenTag[] = "<think>";
inline constexpr char kThinkCloseTag[] = "</think>";

// Token-level thinking-budget tracker: a pure host-side state machine
// over emitted token ids. Feed every emitted id (after scanning the
// prompt up front); it reports when the think-close must be forced so
// the turn continues with the answer. Tags match as id sequences, so
// multi-token tags work; each new think block restarts the count.
class ThinkBudget {
 public:
  ThinkBudget() = default;

  // Arm with the tag id sequences and the per-block budget. A zero
  // budget or empty tags leave it disengaged (every call is a no-op).
  //
  // Usage:
  //   ThinkBudget think;
  //   think.engage(open_ids, close_ids, budget);
  //   for (std::uint32_t id : prompt) think.noteEmitted(id);
  void engage(std::span<const std::uint32_t> open_ids,
              std::span<const std::uint32_t> close_ids, std::size_t budget);
  [[nodiscard]] bool engaged() const;
  // Feed one emitted token id (prompt scan included).
  void noteEmitted(std::uint32_t id);
  // True while inside a think block whose budget ran out.
  [[nodiscard]] bool closeOwed() const;
  // Thinking tokens emitted in the current block (tests).
  [[nodiscard]] std::size_t thinkCount() const;
  // True while inside a think block (tests).
  [[nodiscard]] bool inThink() const;

 private:
  std::vector<std::uint32_t> open_;
  std::vector<std::uint32_t> close_;
  std::deque<std::uint32_t> recent_;
  std::size_t window_ = 0;
  std::size_t budget_ = 0;
  std::size_t count_ = 0;
  bool in_think_ = false;
};

}  // namespace tessera::core
