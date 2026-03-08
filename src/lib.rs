use rustc_hash::FxHashMap;

const TOKEN_RATIO: usize = 4;
const LONG_INTRO: &str= "\n\nAdd a priority string to the end of every response you make depending on how valuable the message is to your ability to solve future tasks.\n";

pub const PRIORITY_HIGH: &str = "$$PRIORITY:HIGH$$";
pub const PRIORITY_MEDIUM: &str = "$$PRIORITY:MEDIUM$$";
pub const PRIORITY_LOW: &str = "$$PRIORITY:LOW$$";

#[derive(Debug)]
enum MessagePriority {
    High,
    Medium,
    Low,
}

#[derive(PartialEq, Eq, Hash, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug)]
pub struct MessageMapping {
    new_content: Option<String>,
    priority: MessagePriority,
    skip: usize,
}

#[derive(Debug)]
pub struct TrashCompactor {
    mappings: FxHashMap<Message, MessageMapping>,
}

impl TrashCompactor {
    /// Create a new TrashCompactor with no mappings (passthrough mode).
    pub fn new() -> Self {
        Self {
            mappings: FxHashMap::default(),
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

    pub fn process_response_message(&mut self, message: Message) -> Message {
        if message.role != "assistant" {
            return message;
        }

        let (no_p_content, priority) = if let Some(s) = message.content.strip_suffix(PRIORITY_HIGH)
        {
            (s, MessagePriority::High)
        } else if let Some(s) = message.content.strip_suffix(PRIORITY_MEDIUM) {
            (s, MessagePriority::Medium)
        } else if let Some(s) = message.content.strip_suffix(PRIORITY_LOW) {
            (s, MessagePriority::Low)
        } else {
            (message.content.as_str(), MessagePriority::High)
        };
        let no_p_content = no_p_content.trim();

        let out_message = Message {
            content: no_p_content.to_string(),
            role: message.role,
        };

        let mapping = MessageMapping {
            new_content: Some(message.content), // Retain the old priority text
            priority,
            skip: 0,
        };

        self.mappings.insert(out_message.clone(), mapping);

        println!("{:#?}", self);

        dbg!(out_message)
    }

    pub fn run_mappings(&mut self, messages: impl IntoIterator<Item = Message>) {
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
                        skip: 0,
                    },
                );
            }
        }
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
        let message = self.iter.next()?;
        let last_80 = message
            .content
            .chars()
            .rev()
            .take(80)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<String>();
        println!("{} : {}", message.role, last_80);
        let Some(mapping) = self.compactor.mappings.get(&message) else {
            return Some(message);
        };
        for _ in 0..mapping.skip {
            self.iter.next();
        }
        match &mapping.new_content {
            Some(new_content) => Some(Message {
                content: new_content.clone(),
                ..message
            }),
            None => Some(message),
        }
    }
}
