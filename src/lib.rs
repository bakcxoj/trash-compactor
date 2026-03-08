use rustc_hash::FxHashMap;
use std::sync::LazyLock;

// ============================================================================
// Tokenizer
// ============================================================================

static BPE: LazyLock<tiktoken_rs::CoreBPE> =
    LazyLock::new(|| tiktoken_rs::o200k_base().expect("failed to initialize o200k_base tokenizer"));

/// Count tokens in text using the o200k_base BPE tokenizer.
pub fn count_tokens(text: &str) -> usize {
    BPE.encode_with_special_tokens(text).len()
}

// ============================================================================
// Compaction Prompts
// ============================================================================

pub const COMPACTION_SYSTEM_PROMPT: &str = "You are a context compactor. Your job is to summarize conversation messages concisely while preserving all important technical details, decisions, code snippets, and context. Maintain factual accuracy. Keep the summary as compact as possible while retaining critical information.";

pub const COMPACTION_USER_PREFIX: &str = "Summarize the following conversation messages into a concise summary that preserves all important technical details, decisions, code snippets, and context. Maintain factual accuracy.\n\n";

// ============================================================================
// Constants
// ============================================================================

const LONG_INTRO: &str = "\n\nAdd a priority string to the end of every response you make depending on how valuable the message is to your ability to solve future tasks.\n";

pub const PRIORITY_HIGH: &str = "$$PRIORITY:HIGH$$";
pub const PRIORITY_MEDIUM: &str = "$$PRIORITY:MEDIUM$$";
pub const PRIORITY_LOW: &str = "$$PRIORITY:LOW$$";

// ============================================================================
// Model Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// How large the context can get before we start compacting (in TOKENS)
    pub max_context_size: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            max_context_size: 30_000,
        }
    }
}

// ============================================================================
// Message Priority
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessagePriority {
    Low,
    Medium,
    High,
}

impl MessagePriority {
    /// Numeric rank for ordering: Low=0, Medium=1, High=2
    fn rank(self) -> u8 {
        match self {
            Self::Low => 0,
            Self::Medium => 1,
            Self::High => 2,
        }
    }
}

impl PartialOrd for MessagePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MessagePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct MessageMapping {
    pub new_content: Option<String>,
    pub priority: MessagePriority,
    pub skip_me: bool,
}

// ============================================================================
// Compaction Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionPhase {
    Phase2MediumLong,
    Phase3MediumAll,
    Phase4High,
}

#[derive(Debug)]
pub struct CompactionPlan {
    pub phase: CompactionPhase,
    pub messages_to_compact: Vec<Message>,
    pub combined_text: String,
    pub survivor_index: usize,
}

#[derive(Debug)]
pub struct CompactionResult {
    pub phase: CompactionPhase,
    pub survivor: Message,
    pub compacted_text: String,
    pub messages_to_skip: Vec<Message>,
    pub promote_to_high: bool,
}

// ============================================================================
// TrashCompactor
// ============================================================================

#[derive(Debug)]
pub struct TrashCompactor {
    mappings: FxHashMap<Message, MessageMapping>,
    model_configs: FxHashMap<String, ModelConfig>,
    compaction_phase_state: Option<CompactionPhase>,
}

impl TrashCompactor {
    /// Create a new TrashCompactor with no mappings (passthrough mode).
    pub fn new() -> Self {
        Self {
            mappings: FxHashMap::default(),
            model_configs: FxHashMap::default(),
            compaction_phase_state: None,
        }
    }

    pub fn compact<I>(&self, messages: I) -> CompactMessageIter<'_, I::IntoIter>
    where
        I: IntoIterator<Item = Message>,
    {
        CompactMessageIter {
            compactor: self,
            iter: messages.into_iter(),
        }
    }

    /// Process an assistant response message to extract priority and update internal mappings.
    /// This is a mutating/side-effect function that stores the mapping for future compaction.
    /// The caller is responsible for stripping priority markers from content before display.
    ///
    /// When an assistant message has a priority, all previous non-system messages in the
    /// conversation that do not already have a mapping will inherit that same priority.
    pub fn process_response_message(&mut self, message: &Message, conversation: &[Message]) {
        if message.role != "assistant" {
            return;
        }

        let (no_p_content, priority) = if let Some(s) = message.content.strip_suffix(PRIORITY_HIGH)
        {
            (s, MessagePriority::High)
        } else if let Some(s) = message.content.strip_suffix(PRIORITY_MEDIUM) {
            (s, MessagePriority::Medium)
        } else if let Some(s) = message.content.strip_suffix(PRIORITY_LOW) {
            (s, MessagePriority::Low)
        } else {
            (message.content.as_str(), MessagePriority::Medium)
        };
        let no_p_content = no_p_content.trim();

        let out_message = Message {
            content: no_p_content.to_string(),
            role: message.role.clone(),
        };

        let mapping = MessageMapping {
            new_content: Some(message.content.clone()), // Retain the old priority text
            priority,
            skip_me: false,
        };

        self.mappings.insert(out_message.clone(), mapping);

        // Propagate priority to previous non-system messages that don't have a mapping.
        // If assistant message is not present in the provided slice (normal runtime flow),
        // treat all provided messages as prior messages.
        let previous_messages = conversation
            .iter()
            .position(|m| m.role == message.role && m.content == message.content)
            .map(|position| &conversation[..position])
            .unwrap_or(conversation);

        for msg in previous_messages {
            // Skip system messages and messages that already have a mapping
            if msg.role != "system" && !self.mappings.contains_key(msg) {
                let propagated_mapping = MessageMapping {
                    new_content: None,
                    priority,
                    skip_me: false,
                };
                self.mappings.insert(msg.clone(), propagated_mapping);
            }
        }
    }

    /// Strip priority marker strings from content text.
    /// Returns the content with any trailing priority markers removed.
    pub fn strip_priority_markers(content: &str) -> String {
        let (no_p_content, _) = if let Some(s) = content.strip_suffix(PRIORITY_HIGH) {
            (s, MessagePriority::High)
        } else if let Some(s) = content.strip_suffix(PRIORITY_MEDIUM) {
            (s, MessagePriority::Medium)
        } else if let Some(s) = content.strip_suffix(PRIORITY_LOW) {
            (s, MessagePriority::Low)
        } else {
            (content, MessagePriority::Medium)
        };
        no_p_content.to_string()
    }

    /// Remove all complete priority marker strings from content text.
    /// Used for user-visible content sanitization (including streaming output).
    pub fn remove_all_priority_markers(content: &str) -> String {
        content
            .replace(PRIORITY_HIGH, "")
            .replace(PRIORITY_MEDIUM, "")
            .replace(PRIORITY_LOW, "")
    }

    /// Returns the length of the longest suffix of `content` that is a proper
    /// prefix of any priority marker constant. Returns 0 if no partial match.
    /// This is used during streaming to hold back content that might be the
    /// start of a marker split across chunks.
    pub fn partial_marker_suffix_len(content: &str) -> usize {
        let markers = [PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW];

        // A full marker suffix is not a partial overlap.
        if markers.iter().any(|marker| content.ends_with(marker)) {
            return 0;
        }

        let mut max_overlap = 0;
        for marker in &markers {
            let marker_bytes = marker.as_bytes();
            // Check prefixes of length 1..marker_len (exclusive of full marker,
            // since full markers are handled by strip_priority_markers)
            for prefix_len in (1..marker_bytes.len()).rev() {
                if content.as_bytes().ends_with(&marker_bytes[..prefix_len]) {
                    max_overlap = max_overlap.max(prefix_len);
                    break; // Found longest prefix match for this marker
                }
            }
        }
        max_overlap
    }

    pub fn run_mappings(&mut self, _model_name: &str, messages: impl IntoIterator<Item = Message>) {
        for message in messages {
            if !self.mappings.contains_key(&message) && message.role == "system" {
                let new_content = [
                    &message.content,
                    LONG_INTRO,
                    PRIORITY_HIGH,
                    " for high value messages.\n",
                    PRIORITY_MEDIUM,
                    " for medium value messages.\n",
                    PRIORITY_LOW,
                    " for low priority messages.\n",
                ]
                .concat();
                self.mappings.insert(
                    message,
                    MessageMapping {
                        new_content: Some(new_content),
                        priority: MessagePriority::High,
                        skip_me: false,
                    },
                );
            }
        }
    }

    // ========================================================================
    // Context Budget Compaction Methods
    // ========================================================================

    /// Get max_context_size for a model (falls back to default).
    pub fn get_max_context_size(&self, model_name: &str) -> usize {
        self.model_configs
            .get(model_name)
            .map(|c| c.max_context_size)
            .unwrap_or(ModelConfig::default().max_context_size)
    }

    /// Count total tokens for messages using current mappings.
    pub fn estimate_context_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|msg| match self.mappings.get(msg) {
                Some(m) if m.skip_me => 0,
                Some(m) => count_tokens(m.new_content.as_deref().unwrap_or(&msg.content)),
                None => count_tokens(&msg.content),
            })
            .sum()
    }

    /// Get effective content for a message (using mapping if available).
    fn get_effective_content(&self, message: &Message) -> String {
        match self.mappings.get(message) {
            Some(m) if m.skip_me => String::new(),
            Some(m) => m.new_content.clone().unwrap_or(message.content.clone()),
            None => message.content.clone(),
        }
    }

    /// Get priority for a message (using mapping if available, defaults to High).
    fn get_priority(&self, message: &Message) -> MessagePriority {
        self.mappings
            .get(message)
            .map(|m| m.priority)
            .unwrap_or(MessagePriority::Medium)
    }
}

/// Calculate preservation value score for a message.
/// Higher score = more likely to be preserved intact (chosen as survivor).
///
/// - recency: position / (total - 1), clamped to [0.0, 1.0]
///   (0.0 = oldest non-system message, 1.0 = newest)
/// - brevity: 1.0 / (1.0 + token_count as f64 / 500.0)
///   (approaches 0 for very long messages, 1.0 for empty)
///
/// score = RECENCY_WEIGHT * recency + LENGTH_WEIGHT * brevity
///
/// Default weights: RECENCY_WEIGHT = 0.6, LENGTH_WEIGHT = 0.4
pub fn message_value_score(position: usize, total_messages: usize, token_count: usize) -> f64 {
    const RECENCY_WEIGHT: f64 = 0.6;
    const LENGTH_WEIGHT: f64 = 0.4;

    let recency = if total_messages <= 1 {
        1.0
    } else {
        position as f64 / (total_messages - 1) as f64
    };

    let brevity = 1.0 / (1.0 + token_count as f64 / 500.0);

    RECENCY_WEIGHT * recency + LENGTH_WEIGHT * brevity
}

impl TrashCompactor {
    /// Check if message is skipped.
    fn is_skipped(&self, message: &Message) -> bool {
        self.mappings
            .get(message)
            .map(|m| m.skip_me)
            .unwrap_or(false)
    }

    /// Phase 1: Skip all Low-priority non-system messages. Returns token count after.
    pub fn skip_low_priority(&mut self, messages: &[Message]) -> usize {
        for message in messages {
            if message.role == "system" {
                continue; // Never skip system messages
            }
            if let Some(mapping) = self.mappings.get_mut(message) {
                if mapping.priority == MessagePriority::Low {
                    mapping.skip_me = true;
                }
            }
        }
        self.estimate_context_tokens(messages)
    }

    /// Calculate token count percentages for each priority level.
    fn calculate_priority_percentages(&self, messages: &[Message]) -> (f64, f64, f64) {
        let mut low_tokens = 0usize;
        let mut medium_tokens = 0usize;
        let mut high_tokens = 0usize;

        for message in messages {
            if self.is_skipped(message) || message.role == "system" {
                continue;
            }
            let tokens = count_tokens(&self.get_effective_content(message));
            match self.get_priority(message) {
                MessagePriority::Low => low_tokens += tokens,
                MessagePriority::Medium => medium_tokens += tokens,
                MessagePriority::High => high_tokens += tokens,
            }
        }

        let total = low_tokens + medium_tokens + high_tokens;
        if total == 0 {
            return (0.0, 0.0, 0.0);
        }

        (
            low_tokens as f64 / total as f64,
            medium_tokens as f64 / total as f64,
            high_tokens as f64 / total as f64,
        )
    }

    /// Compute the next compaction plan, or None if no more phases apply.
    pub fn plan_compaction(
        &self,
        messages: &[Message],
        max_context_size: usize,
    ) -> Option<CompactionPlan> {
        let current_tokens = self.estimate_context_tokens(messages);
        if current_tokens <= max_context_size {
            return None;
        }

        // Calculate percentages before determining phase
        let (low_pct, medium_pct, _) = self.calculate_priority_percentages(messages);

        match self.compaction_phase_state {
            None => {
                // Entry point: decide between Phase 2 or skip to Phase 3
                if low_pct < 0.05 {
                    // Phase 2: Long medium messages
                    self.plan_phase2(messages)
                } else {
                    // Skip Phase 2, go directly to Phase 3
                    self.plan_phase3(messages)
                }
            }
            Some(CompactionPhase::Phase2MediumLong) => {
                // After Phase 2, go to Phase 3
                self.plan_phase3(messages)
            }
            Some(CompactionPhase::Phase3MediumAll) => {
                // After Phase 3, check medium percentage
                if medium_pct < 0.30 {
                    // Skip to Phase 4 (medium was already small)
                    self.plan_phase4(messages)
                } else {
                    self.plan_phase4(messages)
                }
            }
            Some(CompactionPhase::Phase4High) => {
                // No more phases
                None
            }
        }
    }

    /// Phase 2: Compact long medium-priority messages (top quartile).
    fn plan_phase2(&self, messages: &[Message]) -> Option<CompactionPlan> {
        // Gather non-system, non-skipped medium messages with their source indices and token counts
        let medium_messages: Vec<(usize, Message, usize)> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != "system" && !self.is_skipped(m))
            .filter(|(_, m)| self.get_priority(m) == MessagePriority::Medium)
            .map(|(index, m)| {
                let tokens = count_tokens(&self.get_effective_content(m));
                (index, m.clone(), tokens)
            })
            .collect();

        if medium_messages.len() < 2 {
            // Not enough for quartile calculation, skip to Phase 3
            return self.plan_phase3(messages);
        }

        // Sort by token count ascending
        let mut sorted = medium_messages.clone();
        sorted.sort_by_key(|(_, _, tokens)| *tokens);

        // Calculate P75 index
        let n = sorted.len();
        let p75_index = ((0.75 * n as f64).ceil() as usize).saturating_sub(1);
        let p75_threshold = sorted[p75_index].2;

        // Select messages with token count >= P75
        let long_medium: Vec<(usize, Message, usize)> = medium_messages
            .into_iter()
            .filter(|(_, _, tokens)| *tokens >= p75_threshold)
            .collect();

        if long_medium.is_empty() {
            return self.plan_phase3(messages);
        }

        // Find survivor (highest value score)
        let survivor_index = long_medium
            .iter()
            .map(|(pos, _, tokens)| (*pos, message_value_score(*pos, messages.len(), *tokens)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pos, _)| pos)
            .unwrap_or(0);

        // Build combined text with separators
        let combined_text = long_medium
            .iter()
            .enumerate()
            .map(|(i, (_, m, _))| {
                format!(
                    "\n---\n[Message {} ({})]\n{}",
                    i,
                    m.role,
                    self.get_effective_content(m)
                )
            })
            .collect::<String>();

        let messages_to_compact = long_medium.into_iter().map(|(_, m, _)| m).collect();

        Some(CompactionPlan {
            phase: CompactionPhase::Phase2MediumLong,
            messages_to_compact,
            combined_text,
            survivor_index,
        })
    }

    /// Phase 3: Compact all medium-priority messages together.
    fn plan_phase3(&self, messages: &[Message]) -> Option<CompactionPlan> {
        let medium_messages: Vec<(usize, Message, usize)> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != "system" && !self.is_skipped(m))
            .filter(|(_, m)| self.get_priority(m) == MessagePriority::Medium)
            .map(|(index, m)| {
                let tokens = count_tokens(&self.get_effective_content(m));
                (index, m.clone(), tokens)
            })
            .collect();

        if medium_messages.is_empty() {
            return self.plan_phase4(messages);
        }

        // Find survivor (highest value score)
        let survivor_index = medium_messages
            .iter()
            .map(|(pos, _, tokens)| (*pos, message_value_score(*pos, messages.len(), *tokens)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pos, _)| pos)
            .unwrap_or(0);

        // Build combined text
        let combined_text = medium_messages
            .iter()
            .enumerate()
            .map(|(i, (_, m, _))| {
                format!(
                    "\n---\n[Message {} ({})]\n{}",
                    i,
                    m.role,
                    self.get_effective_content(m)
                )
            })
            .collect::<String>();

        let messages_to_compact = medium_messages.into_iter().map(|(_, m, _)| m).collect();

        Some(CompactionPlan {
            phase: CompactionPhase::Phase3MediumAll,
            messages_to_compact,
            combined_text,
            survivor_index,
        })
    }

    /// Phase 4: Compact high-priority messages (including Phase 3 survivor if promoted).
    fn plan_phase4(&self, messages: &[Message]) -> Option<CompactionPlan> {
        let high_messages: Vec<(usize, Message, usize)> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.role != "system" && !self.is_skipped(m))
            .filter(|(_, m)| self.get_priority(m) == MessagePriority::High)
            .map(|(index, m)| {
                let tokens = count_tokens(&self.get_effective_content(m));
                (index, m.clone(), tokens)
            })
            .collect();

        if high_messages.is_empty() {
            return None;
        }

        // Find survivor (highest value score)
        let survivor_index = high_messages
            .iter()
            .map(|(pos, _, tokens)| (*pos, message_value_score(*pos, messages.len(), *tokens)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(pos, _)| pos)
            .unwrap_or(0);

        // Build combined text
        let combined_text = high_messages
            .iter()
            .enumerate()
            .map(|(i, (_, m, _))| {
                format!(
                    "\n---\n[Message {} ({})]\n{}",
                    i,
                    m.role,
                    self.get_effective_content(m)
                )
            })
            .collect::<String>();

        let messages_to_compact = high_messages.into_iter().map(|(_, m, _)| m).collect();

        Some(CompactionPlan {
            phase: CompactionPhase::Phase4High,
            messages_to_compact,
            combined_text,
            survivor_index,
        })
    }

    /// Apply compaction results back to mappings.
    pub fn apply_compaction(&mut self, result: CompactionResult) {
        // Set survivor's new_content
        if let Some(mapping) = self.mappings.get_mut(&result.survivor) {
            mapping.new_content = Some(result.compacted_text.clone());
            if result.promote_to_high {
                mapping.priority = MessagePriority::High;
            }
        } else {
            // Create mapping if it doesn't exist
            self.mappings.insert(
                result.survivor.clone(),
                MessageMapping {
                    new_content: Some(result.compacted_text),
                    priority: if result.promote_to_high {
                        MessagePriority::High
                    } else {
                        MessagePriority::Medium
                    },
                    skip_me: false,
                },
            );
        }

        // Set skip_me on all non-survivor messages
        for msg in result.messages_to_skip {
            // Never skip system messages
            if msg.role == "system" {
                continue;
            }
            if let Some(mapping) = self.mappings.get_mut(&msg) {
                mapping.skip_me = true;
            } else {
                self.mappings.insert(
                    msg,
                    MessageMapping {
                        new_content: None,
                        priority: MessagePriority::Low,
                        skip_me: true,
                    },
                );
            }
        }

        // Advance phase state
        self.compaction_phase_state = Some(result.phase);
    }

    /// Reset compaction phase state for a new request.
    pub fn reset_compaction_state(&mut self) {
        self.compaction_phase_state = None;
    }
}

impl Default for TrashCompactor {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CompactMessageIter<'a, I>
where
    I: Iterator<Item = Message>,
{
    compactor: &'a TrashCompactor,
    iter: I,
}

impl<'a, I> Iterator for CompactMessageIter<'a, I>
where
    I: Iterator<Item = Message>,
{
    type Item = Message;

    fn next(&mut self) -> Option<Self::Item> {
        let (message, mapping) = self.iter.find_map(|message| {
            if let Some(mapping) = self.compactor.mappings.get(&message) {
                if mapping.skip_me {
                    None
                } else {
                    Some((message, Some(mapping)))
                }
            } else {
                Some((message, None))
            }
        })?;
        let Some(mapping) = mapping else {
            return Some(message);
        };
        match &mapping.new_content {
            Some(new_content) => Some(Message {
                content: new_content.clone(),
                ..message
            }),
            None => Some(message),
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_tokens_empty() {
        assert_eq!(count_tokens(""), 0);
    }

    #[test]
    fn test_count_tokens_simple() {
        // "hello world" should have a reasonable token count (not chars/4)
        let tokens = count_tokens("hello world");
        assert!(tokens > 0);
        assert!(tokens < 20); // Should be around 2-3 tokens
    }

    #[test]
    fn test_estimate_context_tokens() {
        let mut compactor = TrashCompactor::new();
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "Hello world".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi there".to_string(),
            },
        ];

        let tokens = compactor.estimate_context_tokens(&messages);
        assert!(tokens > 0);

        // Test with skipped message
        compactor.mappings.insert(
            messages[1].clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Low,
                skip_me: true,
            },
        );

        let tokens_after_skip = compactor.estimate_context_tokens(&messages);
        assert!(tokens_after_skip < tokens);
    }

    #[test]
    fn test_skip_low_priority_never_skips_system() {
        let mut compactor = TrashCompactor::new();

        let messages = vec![
            Message {
                role: "system".to_string(),
                content: "You are helpful".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Hello".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Hi".to_string(),
            },
        ];

        // Set up mappings with different priorities
        compactor.mappings.insert(
            messages[0].clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Low, // System with Low priority
                skip_me: false,
            },
        );
        compactor.mappings.insert(
            messages[1].clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Low,
                skip_me: false,
            },
        );
        compactor.mappings.insert(
            messages[2].clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::High,
                skip_me: false,
            },
        );

        compactor.skip_low_priority(&messages);

        // System message should NOT be skipped
        assert!(!compactor.mappings.get(&messages[0]).unwrap().skip_me);
        // User message with Low priority should be skipped
        assert!(compactor.mappings.get(&messages[1]).unwrap().skip_me);
        // High priority message should not be skipped
        assert!(!compactor.mappings.get(&messages[2]).unwrap().skip_me);
    }

    #[test]
    fn test_plan_compaction_phase2_p75_threshold() {
        let mut compactor = TrashCompactor::new();

        // Create 4 medium messages with different token counts
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "x".repeat(100), // 100 tokens (roughly)
            },
            Message {
                role: "assistant".to_string(),
                content: "y".repeat(200), // ~200 tokens
            },
            Message {
                role: "user".to_string(),
                content: "z".repeat(300), // ~300 tokens
            },
            Message {
                role: "assistant".to_string(),
                content: "w".repeat(400), // ~400 tokens
            },
        ];

        // All medium priority
        for msg in &messages {
            compactor.mappings.insert(
                msg.clone(),
                MessageMapping {
                    new_content: None,
                    priority: MessagePriority::Medium,
                    skip_me: false,
                },
            );
        }

        // With 4 messages sorted by token count: [100, 200, 300, 400]
        // P75 index = ceil(0.75 * 4) - 1 = 3 - 1 = 2
        // P75 threshold = 300
        // Messages >= 300: indices 2 and 3

        let plan = compactor.plan_phase2(&messages).unwrap();
        assert_eq!(plan.phase, CompactionPhase::Phase2MediumLong);
        assert!(plan.messages_to_compact.len() >= 2);
    }

    #[test]
    fn test_apply_compaction_promotes_to_high() {
        let mut compactor = TrashCompactor::new();

        let survivor = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let other = Message {
            role: "assistant".to_string(),
            content: "Hi".to_string(),
        };

        compactor.mappings.insert(
            survivor.clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Medium,
                skip_me: false,
            },
        );
        compactor.mappings.insert(
            other.clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Medium,
                skip_me: false,
            },
        );

        let result = CompactionResult {
            phase: CompactionPhase::Phase3MediumAll,
            survivor: survivor.clone(),
            compacted_text: "Compacted content".to_string(),
            messages_to_skip: vec![other.clone()],
            promote_to_high: true,
        };

        compactor.apply_compaction(result);

        // Survivor should have new content and be promoted to High
        let survivor_mapping = compactor.mappings.get(&survivor).unwrap();
        assert_eq!(
            survivor_mapping.new_content,
            Some("Compacted content".to_string())
        );
        assert_eq!(survivor_mapping.priority, MessagePriority::High);
        assert!(!survivor_mapping.skip_me);

        // Other should be skipped
        let other_mapping = compactor.mappings.get(&other).unwrap();
        assert!(other_mapping.skip_me);
    }

    #[test]
    fn test_apply_compaction_never_skips_system() {
        let mut compactor = TrashCompactor::new();

        let survivor = Message {
            role: "user".to_string(),
            content: "Hello".to_string(),
        };
        let system_msg = Message {
            role: "system".to_string(),
            content: "System prompt".to_string(),
        };

        compactor.mappings.insert(
            survivor.clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Medium,
                skip_me: false,
            },
        );
        compactor.mappings.insert(
            system_msg.clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::High,
                skip_me: false,
            },
        );

        let result = CompactionResult {
            phase: CompactionPhase::Phase3MediumAll,
            survivor: survivor.clone(),
            compacted_text: "Compacted".to_string(),
            messages_to_skip: vec![system_msg.clone()], // Try to skip system
            promote_to_high: false,
        };

        compactor.apply_compaction(result);

        // System message should NOT be skipped
        let system_mapping = compactor.mappings.get(&system_msg).unwrap();
        assert!(!system_mapping.skip_me);
    }

    #[test]
    fn test_compaction_phase_progression() {
        let mut compactor = TrashCompactor::new();

        let messages = vec![Message {
            role: "user".to_string(),
            content: "Test".to_string(),
        }];

        compactor.mappings.insert(
            messages[0].clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::Medium,
                skip_me: false,
            },
        );

        // Start with no phase completed
        assert!(compactor.compaction_phase_state.is_none());

        // Apply Phase 2 result
        let result = CompactionResult {
            phase: CompactionPhase::Phase2MediumLong,
            survivor: messages[0].clone(),
            compacted_text: "Compacted".to_string(),
            messages_to_skip: vec![],
            promote_to_high: false,
        };
        compactor.apply_compaction(result);
        assert_eq!(
            compactor.compaction_phase_state,
            Some(CompactionPhase::Phase2MediumLong)
        );

        // Apply Phase 3 result
        let result = CompactionResult {
            phase: CompactionPhase::Phase3MediumAll,
            survivor: messages[0].clone(),
            compacted_text: "Compacted2".to_string(),
            messages_to_skip: vec![],
            promote_to_high: true,
        };
        compactor.apply_compaction(result);
        assert_eq!(
            compactor.compaction_phase_state,
            Some(CompactionPhase::Phase3MediumAll)
        );

        // Apply Phase 4 result
        let result = CompactionResult {
            phase: CompactionPhase::Phase4High,
            survivor: messages[0].clone(),
            compacted_text: "Compacted3".to_string(),
            messages_to_skip: vec![],
            promote_to_high: false,
        };
        compactor.apply_compaction(result);
        assert_eq!(
            compactor.compaction_phase_state,
            Some(CompactionPhase::Phase4High)
        );

        // After Phase 4, plan_compaction should return None
        compactor.mappings.get_mut(&messages[0]).unwrap().priority = MessagePriority::High;
        let plan = compactor.plan_compaction(&messages, 0);
        assert!(plan.is_none());
    }

    #[test]
    fn test_reset_compaction_state() {
        let mut compactor = TrashCompactor::new();
        compactor.compaction_phase_state = Some(CompactionPhase::Phase4High);

        compactor.reset_compaction_state();
        assert!(compactor.compaction_phase_state.is_none());
    }

    #[test]
    fn test_strip_priority_markers() {
        // Test HIGH priority marker
        let content = "Hello world $$PRIORITY:HIGH$$";
        assert_eq!(
            TrashCompactor::strip_priority_markers(content),
            "Hello world "
        );

        // Test MEDIUM priority marker
        let content = "Hello world $$PRIORITY:MEDIUM$$";
        assert_eq!(
            TrashCompactor::strip_priority_markers(content),
            "Hello world "
        );

        // Test LOW priority marker
        let content = "Hello world $$PRIORITY:LOW$$";
        assert_eq!(
            TrashCompactor::strip_priority_markers(content),
            "Hello world "
        );

        // Test no priority marker
        let content = "Hello world";
        assert_eq!(
            TrashCompactor::strip_priority_markers(content),
            "Hello world"
        );

        // Test with extra whitespace before marker (must be preserved)
        let content = "Hello world   $$PRIORITY:HIGH$$";
        assert_eq!(
            TrashCompactor::strip_priority_markers(content),
            "Hello world   "
        );

        // Test with no marker (must preserve all whitespace exactly)
        let content = "  I need to...\n\n";
        assert_eq!(
            TrashCompactor::strip_priority_markers(content),
            "  I need to...\n\n"
        );

        // Streaming-style chunk with leading space should stay intact
        let content = " need";
        assert_eq!(TrashCompactor::strip_priority_markers(content), " need");

        // Test empty string
        let content = "";
        assert_eq!(TrashCompactor::strip_priority_markers(content), "");

        // Test only marker
        let content = "$$PRIORITY:HIGH$$";
        assert_eq!(TrashCompactor::strip_priority_markers(content), "");
    }

    #[test]
    fn test_remove_all_priority_markers() {
        let content = "Hello $$PRIORITY:HIGH$$ world $$PRIORITY:LOW$$!";
        assert_eq!(
            TrashCompactor::remove_all_priority_markers(content),
            "Hello  world !"
        );

        let content = "$$PRIORITY:MEDIUM$$";
        assert_eq!(TrashCompactor::remove_all_priority_markers(content), "");
    }

    #[test]
    fn test_partial_marker_suffix_len_no_match() {
        assert_eq!(TrashCompactor::partial_marker_suffix_len("Hello world"), 0);
    }

    #[test]
    fn test_partial_marker_suffix_len_dollar_sign() {
        // "$" is a 1-byte prefix of "$$PRIORITY:..."
        assert_eq!(TrashCompactor::partial_marker_suffix_len("Hello $"), 1);
    }

    #[test]
    fn test_partial_marker_suffix_len_double_dollar() {
        // "$$" is a 2-byte prefix
        assert_eq!(TrashCompactor::partial_marker_suffix_len("Hello $$"), 2);
    }

    #[test]
    fn test_partial_marker_suffix_len_partial_priority() {
        // "$$PRIORITY:" is the 11-byte common prefix
        assert_eq!(
            TrashCompactor::partial_marker_suffix_len("Hello $$PRIORITY:"),
            11
        );
    }

    #[test]
    fn test_partial_marker_suffix_len_almost_complete() {
        // "$$PRIORITY:HIG" is 14 bytes (HIGH$$ minus "H$$")
        assert_eq!(
            TrashCompactor::partial_marker_suffix_len("Hello $$PRIORITY:HIG"),
            14
        );
    }

    #[test]
    fn test_partial_marker_suffix_len_full_marker_returns_zero() {
        // Full marker is NOT a proper prefix; handled by strip
        assert_eq!(
            TrashCompactor::partial_marker_suffix_len("Hello $$PRIORITY:HIGH$$"),
            0
        );
    }

    #[test]
    fn test_partial_marker_suffix_len_medium_partial() {
        // "$$PRIORITY:MED" is 14 bytes
        assert_eq!(
            TrashCompactor::partial_marker_suffix_len("Hello $$PRIORITY:MED"),
            14
        );
    }

    #[test]
    fn test_partial_marker_suffix_len_low_partial() {
        // "$$PRIORITY:LO" is 13 bytes
        assert_eq!(
            TrashCompactor::partial_marker_suffix_len("Hello $$PRIORITY:LO"),
            13
        );
    }

    #[test]
    fn test_partial_marker_suffix_len_empty() {
        assert_eq!(TrashCompactor::partial_marker_suffix_len(""), 0);
    }

    #[test]
    fn test_partial_marker_suffix_len_just_marker_prefix() {
        // "$$PRIORITY:HIGH$" is 16 bytes (one "$" short of complete)
        assert_eq!(
            TrashCompactor::partial_marker_suffix_len("$$PRIORITY:HIGH$"),
            16
        );
    }

    #[test]
    fn test_process_response_message_no_return() {
        let mut compactor = TrashCompactor::new();

        // Test assistant message with priority marker
        let message = Message {
            role: "assistant".to_string(),
            content: "Response text $$PRIORITY:MEDIUM$$".to_string(),
        };
        // Pass empty conversation since we're just testing the basic mapping creation
        compactor.process_response_message(&message, &[]);

        // Verify mapping was created (stripped content as key)
        let stripped_message = Message {
            role: "assistant".to_string(),
            content: "Response text".to_string(),
        };
        assert!(compactor.mappings.contains_key(&stripped_message));

        let mapping = compactor.mappings.get(&stripped_message).unwrap();
        assert_eq!(mapping.priority, MessagePriority::Medium);
        assert_eq!(
            mapping.new_content,
            Some("Response text $$PRIORITY:MEDIUM$$".to_string())
        );
        assert!(!mapping.skip_me);

        // Test non-assistant message (should be ignored)
        let user_message = Message {
            role: "user".to_string(),
            content: "User text".to_string(),
        };
        compactor.process_response_message(&user_message, &[]);

        // User message should not be in mappings
        let user_key = Message {
            role: "user".to_string(),
            content: "User text".to_string(),
        };
        assert!(!compactor.mappings.contains_key(&user_key));
    }

    #[test]
    fn test_propagate_priority_to_previous_user_messages() {
        let mut compactor = TrashCompactor::new();

        // Create a conversation with user messages followed by assistant
        let conversation = vec![
            Message {
                role: "user".to_string(),
                content: "First user message".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "Second user message".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Assistant response $$PRIORITY:HIGH$$".to_string(),
            },
        ];

        let assistant_msg = &conversation[2];
        compactor.process_response_message(assistant_msg, &conversation);

        // Both user messages should now have High priority
        let first_mapping = compactor.mappings.get(&conversation[0]).unwrap();
        assert_eq!(first_mapping.priority, MessagePriority::High);
        assert_eq!(first_mapping.new_content, None);
        assert!(!first_mapping.skip_me);

        let second_mapping = compactor.mappings.get(&conversation[1]).unwrap();
        assert_eq!(second_mapping.priority, MessagePriority::High);
        assert_eq!(second_mapping.new_content, None);
    }

    #[test]
    fn test_propagate_priority_skips_system_messages() {
        let mut compactor = TrashCompactor::new();

        let conversation = vec![
            Message {
                role: "system".to_string(),
                content: "System prompt".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "User message".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Assistant response $$PRIORITY:LOW$$".to_string(),
            },
        ];

        let assistant_msg = &conversation[2];
        compactor.process_response_message(assistant_msg, &conversation);

        // System message should NOT have a mapping
        assert!(!compactor.mappings.contains_key(&conversation[0]));

        // User message should have Low priority
        let user_mapping = compactor.mappings.get(&conversation[1]).unwrap();
        assert_eq!(user_mapping.priority, MessagePriority::Low);
    }

    #[test]
    fn test_propagate_priority_skips_already_mapped_messages() {
        let mut compactor = TrashCompactor::new();

        // Pre-create a mapping for the first user message with High priority
        let user_msg = Message {
            role: "user".to_string(),
            content: "User message".to_string(),
        };
        compactor.mappings.insert(
            user_msg.clone(),
            MessageMapping {
                new_content: None,
                priority: MessagePriority::High,
                skip_me: false,
            },
        );

        let conversation = vec![
            user_msg,
            Message {
                role: "assistant".to_string(),
                content: "Assistant response $$PRIORITY:LOW$$".to_string(),
            },
        ];

        let assistant_msg = &conversation[1];
        compactor.process_response_message(assistant_msg, &conversation);

        // User message should retain its original High priority (not overwritten)
        let user_mapping = compactor.mappings.get(&conversation[0]).unwrap();
        assert_eq!(user_mapping.priority, MessagePriority::High);
    }

    #[test]
    fn test_propagate_priority_to_tool_messages() {
        let mut compactor = TrashCompactor::new();

        let conversation = vec![
            Message {
                role: "user".to_string(),
                content: "User message".to_string(),
            },
            Message {
                role: "tool".to_string(),
                content: "Tool response".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Assistant response $$PRIORITY:HIGH$$".to_string(),
            },
        ];

        let assistant_msg = &conversation[2];
        compactor.process_response_message(assistant_msg, &conversation);

        // Both user and tool messages should have High priority
        let user_mapping = compactor.mappings.get(&conversation[0]).unwrap();
        assert_eq!(user_mapping.priority, MessagePriority::High);

        let tool_mapping = compactor.mappings.get(&conversation[1]).unwrap();
        assert_eq!(tool_mapping.priority, MessagePriority::High);
    }

    #[test]
    fn test_propagate_priority_only_before_assistant() {
        let mut compactor = TrashCompactor::new();

        let conversation = vec![
            Message {
                role: "assistant".to_string(),
                content: "First assistant $$PRIORITY:HIGH$$".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "User message after".to_string(),
            },
        ];

        let first_assistant = &conversation[0];
        compactor.process_response_message(first_assistant, &conversation);

        // User message comes AFTER the assistant, so it should NOT have a mapping
        assert!(!compactor.mappings.contains_key(&conversation[1]));
    }

    #[test]
    fn test_propagate_priority_default_medium_when_no_marker() {
        let mut compactor = TrashCompactor::new();

        let conversation = vec![
            Message {
                role: "user".to_string(),
                content: "User message".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "Assistant response without marker".to_string(),
            },
        ];

        let assistant_msg = &conversation[1];
        compactor.process_response_message(assistant_msg, &conversation);

        // User message should get default Medium priority
        let user_mapping = compactor.mappings.get(&conversation[0]).unwrap();
        assert_eq!(user_mapping.priority, MessagePriority::Medium);
    }

    #[test]
    fn test_propagate_priority_empty_conversation() {
        let mut compactor = TrashCompactor::new();

        let message = Message {
            role: "assistant".to_string(),
            content: "Response $$PRIORITY:HIGH$$".to_string(),
        };

        // Should not panic with empty conversation
        compactor.process_response_message(&message, &[]);

        // Verify the assistant mapping was still created
        let stripped_message = Message {
            role: "assistant".to_string(),
            content: "Response".to_string(),
        };
        assert!(compactor.mappings.contains_key(&stripped_message));
    }

    #[test]
    fn test_propagate_priority_when_assistant_not_in_conversation_slice() {
        let mut compactor = TrashCompactor::new();

        // Runtime call sites pass prior/incoming messages only (no assistant yet).
        let conversation = vec![
            Message {
                role: "system".to_string(),
                content: "System prompt".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "User message".to_string(),
            },
            Message {
                role: "tool".to_string(),
                content: "Tool result".to_string(),
            },
        ];

        let assistant = Message {
            role: "assistant".to_string(),
            content: "Assistant response $$PRIORITY:HIGH$$".to_string(),
        };

        compactor.process_response_message(&assistant, &conversation);

        // System message should still be skipped.
        assert!(!compactor.mappings.contains_key(&conversation[0]));

        // Non-system prior messages should inherit the assistant priority.
        let user_mapping = compactor.mappings.get(&conversation[1]).unwrap();
        assert_eq!(user_mapping.priority, MessagePriority::High);
        assert_eq!(user_mapping.new_content, None);

        let tool_mapping = compactor.mappings.get(&conversation[2]).unwrap();
        assert_eq!(tool_mapping.priority, MessagePriority::High);
        assert_eq!(tool_mapping.new_content, None);
    }

    // ========================================================================
    // Value Score Tests
    // ========================================================================

    #[test]
    fn test_message_value_score_newer_scores_higher() {
        // Two messages at positions 0 and 9 (total=10), same token count
        let score_old = message_value_score(0, 10, 100);
        let score_new = message_value_score(9, 10, 100);
        assert!(score_new > score_old, "Newer message should score higher");
    }

    #[test]
    fn test_message_value_score_shorter_scores_higher() {
        // Two messages at same position, one with 50 tokens and one with 2000 tokens
        let score_short = message_value_score(5, 10, 50);
        let score_long = message_value_score(5, 10, 2000);
        assert!(
            score_short > score_long,
            "Shorter message should score higher"
        );
    }

    #[test]
    fn test_message_value_score_combined() {
        // Newer+shorter beats older+longer
        let score_old_long = message_value_score(0, 10, 2000);
        let score_new_short = message_value_score(9, 10, 50);
        assert!(
            score_new_short > score_old_long,
            "Newer+shorter should beat older+longer"
        );

        // Test the crossover: very old but very short vs very new but very long
        let score_very_old_short = message_value_score(0, 10, 10);
        let score_very_new_long = message_value_score(9, 10, 5000);

        // With 60/40 weighting, recency has more weight
        // recency: 0 vs 1, brevity: ~1.0 vs ~0.09
        // score_old = 0.6 * 0.0 + 0.4 * 0.98 ≈ 0.39
        // score_new = 0.6 * 1.0 + 0.4 * 0.09 ≈ 0.64
        assert!(
            score_very_new_long > score_very_old_short,
            "With 60/40 weighting, very new+long should beat very old+short"
        );
    }

    #[test]
    fn test_message_value_score_single_message() {
        // With total_messages=1, recency should be 1.0 (not divide-by-zero)
        let score = message_value_score(0, 1, 100);
        assert!(score.is_finite(), "Score should be finite");
        assert!(score > 0.0, "Score should be positive");
    }

    #[test]
    fn test_message_value_score_zero_tokens() {
        // Token count of 0 should give brevity of 1.0
        let score = message_value_score(5, 10, 0);
        // recency = 5/9 ≈ 0.556, brevity = 1.0
        // score = 0.6 * 0.556 + 0.4 * 1.0 ≈ 0.733
        assert!(score > 0.7, "Zero tokens should give high brevity score");
    }

    #[test]
    fn test_phase2_survivor_prefers_newer_shorter() {
        let mut compactor = TrashCompactor::new();

        // Create 4 medium messages where the Phase 2 long-message candidates are
        // indices 2 and 3, and index 3 is newer + shorter than index 2.
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "a ".repeat(20),
            },
            Message {
                role: "assistant".to_string(),
                content: "b ".repeat(40),
            },
            Message {
                role: "user".to_string(),
                content: "c ".repeat(700), // Long candidate
            },
            Message {
                role: "assistant".to_string(),
                content: "d ".repeat(600), // Newer + slightly shorter long candidate
            },
        ];

        // All medium priority
        for msg in &messages {
            compactor.mappings.insert(
                msg.clone(),
                MessageMapping {
                    new_content: None,
                    priority: MessagePriority::Medium,
                    skip_me: false,
                },
            );
        }

        let plan = compactor.plan_phase2(&messages).unwrap();

        // The newest message (index 3) should be the survivor because it's
        // both newer and shorter than the other candidates
        assert_eq!(
            plan.survivor_index, 3,
            "Newest/shortest message should be survivor"
        );
    }

    #[test]
    fn test_phase3_survivor_prefers_newer() {
        let mut compactor = TrashCompactor::new();

        // Create 3 medium messages of equal length at different positions.
        // Messages at indices 0 and 2 are exact duplicates (same role + content)
        // to verify survivor scoring uses true source index, not equality lookup.
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "test content".to_string(),
            },
            Message {
                role: "assistant".to_string(),
                content: "test content".to_string(),
            },
            Message {
                role: "user".to_string(),
                content: "test content".to_string(),
            },
        ];

        // All medium priority
        for msg in &messages {
            compactor.mappings.insert(
                msg.clone(),
                MessageMapping {
                    new_content: None,
                    priority: MessagePriority::Medium,
                    skip_me: false,
                },
            );
        }

        let plan = compactor.plan_phase3(&messages).unwrap();

        // The last (newest) message should be the survivor
        assert_eq!(plan.survivor_index, 2, "Newest message should be survivor");
    }

    #[test]
    fn test_phase4_survivor_prefers_newer_shorter() {
        let mut compactor = TrashCompactor::new();

        // Create 3 high-priority messages
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "x".repeat(1000), // Oldest, longest
            },
            Message {
                role: "assistant".to_string(),
                content: "y".repeat(500),
            },
            Message {
                role: "user".to_string(),
                content: "z".repeat(100), // Newest, shortest
            },
        ];

        // All high priority
        for msg in &messages {
            compactor.mappings.insert(
                msg.clone(),
                MessageMapping {
                    new_content: None,
                    priority: MessagePriority::High,
                    skip_me: false,
                },
            );
        }

        let plan = compactor.plan_phase4(&messages).unwrap();

        // The newest message (index 2) should be the survivor
        assert_eq!(
            plan.survivor_index, 2,
            "Newest/shortest message should be survivor"
        );
    }

    #[test]
    fn test_survivor_selection_old_behavior_changed() {
        let mut compactor = TrashCompactor::new();

        // Create messages where the first message is the longest
        let messages = vec![
            Message {
                role: "user".to_string(),
                content: "a".repeat(2000), // Oldest, longest - would have been survivor before
            },
            Message {
                role: "assistant".to_string(),
                content: "b".repeat(100),
            },
            Message {
                role: "user".to_string(),
                content: "c".repeat(100),
            },
            Message {
                role: "assistant".to_string(),
                content: "d".repeat(100), // Newest, shorter
            },
        ];

        // All medium priority
        for msg in &messages {
            compactor.mappings.insert(
                msg.clone(),
                MessageMapping {
                    new_content: None,
                    priority: MessagePriority::Medium,
                    skip_me: false,
                },
            );
        }

        let plan = compactor.plan_phase3(&messages).unwrap();

        // The first message should NOT be chosen as survivor
        assert_ne!(
            plan.survivor_index, 0,
            "First/longest message should NOT be survivor"
        );
        // Instead, the newest message should be survivor
        assert_eq!(plan.survivor_index, 3, "Newest message should be survivor");
    }

    // ========================================================================
    // Streaming Marker Leak Tests
    // ========================================================================

    #[test]
    fn test_streaming_marker_not_split_across_chunks() {
        // Simulate 3 chunks: "Hello ", "world$$PRIORITY:HIGH$$", done
        let chunks = ["Hello ", "world$$PRIORITY:HIGH$$"];
        let mut accumulated = String::new();
        let mut emitted_len = 0usize;
        let mut output = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            accumulated.push_str(chunk);
            let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
            let is_done = i == chunks.len() - 1;
            let holdback = if is_done {
                0
            } else {
                TrashCompactor::partial_marker_suffix_len(&accumulated)
            };
            let safe_len = stripped.len().saturating_sub(holdback);
            if safe_len > emitted_len {
                output.push_str(&stripped[emitted_len..safe_len]);
                emitted_len = safe_len;
            }
        }

        assert_eq!(output, "Hello world");
    }

    #[test]
    fn test_streaming_marker_split_across_two_chunks() {
        // Simulate: "Hello $$PRIORITY:", "HIGH$$", done
        let chunks = ["Hello $$PRIORITY:", "HIGH$$"];
        let mut accumulated = String::new();
        let mut emitted_len = 0usize;
        let mut output = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            accumulated.push_str(chunk);
            let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
            let is_done = i == chunks.len() - 1;
            let holdback = if is_done {
                0
            } else {
                TrashCompactor::partial_marker_suffix_len(&accumulated)
            };
            let safe_len = stripped.len().saturating_sub(holdback);
            if safe_len > emitted_len {
                output.push_str(&stripped[emitted_len..safe_len]);
                emitted_len = safe_len;
            }
        }

        assert_eq!(output, "Hello ");
    }

    #[test]
    fn test_streaming_marker_split_at_every_byte() {
        // For marker "$$PRIORITY:HIGH$$", split it one byte at a time appended to "Hello "
        let marker = "$$PRIORITY:HIGH$$";
        let prefix = "Hello ";

        for split_point in 0..=marker.len() {
            let mut chunks = vec![];

            // First chunk: prefix + first part of marker
            chunks.push(format!("{}{}", prefix, &marker[..split_point]));

            // Second chunk: rest of marker
            if split_point < marker.len() {
                chunks.push(marker[split_point..].to_string());
            }

            let mut accumulated = String::new();
            let mut emitted_len = 0usize;
            let mut output = String::new();

            for (i, chunk) in chunks.iter().enumerate() {
                accumulated.push_str(chunk);
                let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
                let is_done = i == chunks.len() - 1;
                let holdback = if is_done {
                    0
                } else {
                    TrashCompactor::partial_marker_suffix_len(&accumulated)
                };
                let safe_len = stripped.len().saturating_sub(holdback);
                if safe_len > emitted_len {
                    output.push_str(&stripped[emitted_len..safe_len]);
                    emitted_len = safe_len;
                }
            }

            assert_eq!(output, "Hello ", "Failed at split point {}", split_point);
        }
    }

    #[test]
    fn test_streaming_no_marker_passes_through() {
        // Simulate: "Hello ", "world", done
        let chunks = ["Hello ", "world"];
        let mut accumulated = String::new();
        let mut emitted_len = 0usize;
        let mut output = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            accumulated.push_str(chunk);
            let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
            let is_done = i == chunks.len() - 1;
            let holdback = if is_done {
                0
            } else {
                TrashCompactor::partial_marker_suffix_len(&accumulated)
            };
            let safe_len = stripped.len().saturating_sub(holdback);
            if safe_len > emitted_len {
                output.push_str(&stripped[emitted_len..safe_len]);
                emitted_len = safe_len;
            }
        }

        assert_eq!(output, "Hello world");
    }

    #[test]
    fn test_streaming_marker_medium_split() {
        // Simulate: "Result $$PRIORITY:MED", "IUM$$", done
        let chunks = ["Result $$PRIORITY:MED", "IUM$$"];
        let mut accumulated = String::new();
        let mut emitted_len = 0usize;
        let mut output = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            accumulated.push_str(chunk);
            let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
            let is_done = i == chunks.len() - 1;
            let holdback = if is_done {
                0
            } else {
                TrashCompactor::partial_marker_suffix_len(&accumulated)
            };
            let safe_len = stripped.len().saturating_sub(holdback);
            if safe_len > emitted_len {
                output.push_str(&stripped[emitted_len..safe_len]);
                emitted_len = safe_len;
            }
        }

        assert_eq!(output, "Result ");
    }

    #[test]
    fn test_streaming_holdback_releases_on_false_alarm() {
        // Simulate: "Price is $", "5.00 today", done
        let chunks = ["Price is $", "5.00 today"];
        let mut accumulated = String::new();
        let mut emitted_len = 0usize;
        let mut output = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            accumulated.push_str(chunk);
            let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
            let is_done = i == chunks.len() - 1;
            let holdback = if is_done {
                0
            } else {
                TrashCompactor::partial_marker_suffix_len(&accumulated)
            };
            let safe_len = stripped.len().saturating_sub(holdback);
            if safe_len > emitted_len {
                output.push_str(&stripped[emitted_len..safe_len]);
                emitted_len = safe_len;
            }
        }

        // The "$" was held back initially but released when it turned out not to be a marker
        assert_eq!(output, "Price is $5.00 today");
    }

    #[test]
    fn test_streaming_empty_chunks_no_duplicate() {
        // Simulate: "Hello", "", "", " world", done
        let chunks = ["Hello", "", "", " world"];
        let mut accumulated = String::new();
        let mut emitted_len = 0usize;
        let mut output = String::new();

        for (i, chunk) in chunks.iter().enumerate() {
            accumulated.push_str(chunk);
            let stripped = TrashCompactor::remove_all_priority_markers(&accumulated);
            let is_done = i == chunks.len() - 1;
            let holdback = if is_done {
                0
            } else {
                TrashCompactor::partial_marker_suffix_len(&accumulated)
            };
            let safe_len = stripped.len().saturating_sub(holdback);
            if safe_len > emitted_len {
                output.push_str(&stripped[emitted_len..safe_len]);
                emitted_len = safe_len;
            }
        }

        assert_eq!(output, "Hello world");
    }
}
