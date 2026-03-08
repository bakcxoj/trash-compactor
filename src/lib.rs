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
    pub fn process_response_message(&mut self, message: &Message) {
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

        println!("{:#?}", self);
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
        // Gather non-system, non-skipped medium messages with their token counts
        let medium_messages: Vec<(Message, usize)> = messages
            .iter()
            .filter(|m| m.role != "system" && !self.is_skipped(m))
            .filter(|m| self.get_priority(m) == MessagePriority::Medium)
            .map(|m| {
                let tokens = count_tokens(&self.get_effective_content(m));
                (m.clone(), tokens)
            })
            .collect();

        if medium_messages.len() < 2 {
            // Not enough for quartile calculation, skip to Phase 3
            return self.plan_phase3(messages);
        }

        // Sort by token count ascending
        let mut sorted = medium_messages.clone();
        sorted.sort_by_key(|(_, tokens)| *tokens);

        // Calculate P75 index
        let n = sorted.len();
        let p75_index = ((0.75 * n as f64).ceil() as usize).saturating_sub(1);
        let p75_threshold = sorted[p75_index].1;

        // Select messages with token count >= P75
        let long_medium: Vec<Message> = medium_messages
            .into_iter()
            .filter(|(_, tokens)| *tokens >= p75_threshold)
            .map(|(m, _)| m)
            .collect();

        if long_medium.is_empty() {
            return self.plan_phase3(messages);
        }

        // Find survivor (first in conversation order)
        let survivor_index = messages
            .iter()
            .position(|m| long_medium.contains(m))
            .unwrap_or(0);

        // Build combined text with separators
        let combined_text = long_medium
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "\n---\n[Message {} ({})]\n{}",
                    i,
                    m.role,
                    self.get_effective_content(m)
                )
            })
            .collect::<String>();

        Some(CompactionPlan {
            phase: CompactionPhase::Phase2MediumLong,
            messages_to_compact: long_medium,
            combined_text,
            survivor_index,
        })
    }

    /// Phase 3: Compact all medium-priority messages together.
    fn plan_phase3(&self, messages: &[Message]) -> Option<CompactionPlan> {
        let medium_messages: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != "system" && !self.is_skipped(m))
            .filter(|m| self.get_priority(m) == MessagePriority::Medium)
            .cloned()
            .collect();

        if medium_messages.is_empty() {
            return self.plan_phase4(messages);
        }

        // Find survivor (first in conversation order)
        let survivor_index = messages
            .iter()
            .position(|m| medium_messages.contains(m))
            .unwrap_or(0);

        // Build combined text
        let combined_text = medium_messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "\n---\n[Message {} ({})]\n{}",
                    i,
                    m.role,
                    self.get_effective_content(m)
                )
            })
            .collect::<String>();

        Some(CompactionPlan {
            phase: CompactionPhase::Phase3MediumAll,
            messages_to_compact: medium_messages,
            combined_text,
            survivor_index,
        })
    }

    /// Phase 4: Compact high-priority messages (including Phase 3 survivor if promoted).
    fn plan_phase4(&self, messages: &[Message]) -> Option<CompactionPlan> {
        let high_messages: Vec<Message> = messages
            .iter()
            .filter(|m| m.role != "system" && !self.is_skipped(m))
            .filter(|m| self.get_priority(m) == MessagePriority::High)
            .cloned()
            .collect();

        if high_messages.is_empty() {
            return None;
        }

        // Find survivor (first in conversation order)
        let survivor_index = messages
            .iter()
            .position(|m| high_messages.contains(m))
            .unwrap_or(0);

        // Build combined text
        let combined_text = high_messages
            .iter()
            .enumerate()
            .map(|(i, m)| {
                format!(
                    "\n---\n[Message {} ({})]\n{}",
                    i,
                    m.role,
                    self.get_effective_content(m)
                )
            })
            .collect::<String>();

        Some(CompactionPlan {
            phase: CompactionPhase::Phase4High,
            messages_to_compact: high_messages,
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
    fn test_process_response_message_no_return() {
        let mut compactor = TrashCompactor::new();

        // Test assistant message with priority marker
        let message = Message {
            role: "assistant".to_string(),
            content: "Response text $$PRIORITY:MEDIUM$$".to_string(),
        };
        compactor.process_response_message(&message);

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
        compactor.process_response_message(&user_message);

        // User message should not be in mappings
        let user_key = Message {
            role: "user".to_string(),
            content: "User text".to_string(),
        };
        assert!(!compactor.mappings.contains_key(&user_key));
    }
}
