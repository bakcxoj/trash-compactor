use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::Parser;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

// ============================================================================
// CLI Arguments
// ============================================================================

#[derive(Parser, Debug)]
#[command(name = "trash-compactor", version, about = "Context-compacting Ollama proxy")]
struct CliArgs {
    /// Path to TOML config file
    #[arg(short, long, default_value = "trash-compactor.toml")]
    config: std::path::PathBuf,

    /// Ollama backend URL (overrides config file)
    #[arg(long)]
    ollama_url: Option<String>,

    /// Server listen address (overrides config file)
    #[arg(long)]
    listen_addr: Option<String>,

    /// Model to use for compaction summarization (overrides config file)
    #[arg(long)]
    compaction_model: Option<String>,

    /// Default max context size in tokens (overrides config file)
    #[arg(long)]
    max_context_size: Option<usize>,
}

// ============================================================================
// Configuration
// ============================================================================

#[derive(Deserialize, Debug, Default)]
struct FileConfig {
    ollama_url: Option<String>,
    listen_addr: Option<String>,
    compaction_model: Option<String>,
    max_context_size: Option<usize>,
    #[serde(default)]
    models: HashMap<String, ModelOverride>,
}

#[derive(Deserialize, Debug, Clone)]
struct ModelOverride {
    max_context_size: Option<usize>,
}

#[derive(Debug, Clone)]
struct AppConfig {
    ollama_url: String,
    listen_addr: String,
    compaction_model: Option<String>,
    default_max_context_size: usize,
    model_overrides: HashMap<String, ModelOverride>,
}

impl AppConfig {
    fn load() -> Self {
        let args = CliArgs::parse();

        // Load config file if it exists
        let file_config = if args.config.exists() {
            match std::fs::read_to_string(&args.config) {
                Ok(contents) => match toml::from_str::<FileConfig>(&contents) {
                    Ok(cfg) => {
                        println!("Loaded config from {}", args.config.display());
                        cfg
                    }
                    Err(e) => {
                        eprintln!("Warning: Failed to parse config file: {}", e);
                        FileConfig::default()
                    }
                },
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to read config file {}: {}",
                        args.config.display(),
                        e
                    );
                    FileConfig::default()
                }
            }
        } else {
            println!("No config file found at {}, using defaults", args.config.display());
            FileConfig::default()
        };

        // Merge with precedence: CLI > file > defaults
        Self {
            ollama_url: args.ollama_url.unwrap_or_else(|| {
                file_config
                    .ollama_url
                    .unwrap_or_else(|| "http://127.0.0.1:11434".to_string())
            }),
            listen_addr: args.listen_addr.unwrap_or_else(|| {
                file_config
                    .listen_addr
                    .unwrap_or_else(|| "127.0.0.1:11435".to_string())
            }),
            compaction_model: args.compaction_model.or(file_config.compaction_model),
            default_max_context_size: args
                .max_context_size
                .or(file_config.max_context_size)
                .unwrap_or(30_000),
            model_overrides: file_config.models,
        }
    }
}

// ============================================================================
// App State
// ============================================================================

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    ollama_url: String,
    compaction_model: Option<String>,
    default_max_context_size: usize,
    model_overrides: HashMap<String, ModelOverride>,
    compactor: Arc<RwLock<trash_compactor::TrashCompactor>>,
}

// ============================================================================
// Helper functions for message extraction
// ============================================================================

/// Extract content as string from OpenAI-style content field.
/// Handles string, array-of-text-parts, null, and other types.
fn extract_content_string(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            // Extract text parts from content array
            arr.iter()
                .filter_map(|item| {
                    item.get("type")
                        .and_then(|t| t.as_str())
                        .filter(|&t| t == "text")
                        .and_then(|_| item.get("text"))
                        .and_then(|t| t.as_str())
                })
                .collect::<Vec<_>>()
                .join(" ")
        }
        Some(Value::Null) | None => "(null)".to_string(),
        Some(other) => format!("{}", other),
    }
}

// ============================================================================
// JSON ↔ Message conversion helpers
// ============================================================================

/// Convert OpenAI-format messages from JSON to TrashCompactor Message structs.
fn json_messages_to_compact(payload: &Value) -> Vec<trash_compactor::Message> {
    let Some(messages) = payload.get("messages").and_then(|m| m.as_array()) else {
        return Vec::new();
    };

    messages
        .iter()
        .map(|msg| trash_compactor::Message {
            role: msg
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("user")
                .to_string(),
            content: extract_content_string(msg.get("content")),
        })
        .collect()
}

/// Convert TrashCompactor Message structs back to JSON format for Ollama.
fn compact_to_json_messages(messages: Vec<trash_compactor::Message>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "role": m.role,
                "content": m.content
            })
        })
        .collect()
}

/// Extract required model from request payload.
fn required_model_from_payload(payload: &Value) -> Result<String, &'static str> {
    match payload.get("model") {
        Some(Value::String(model)) if !model.trim().is_empty() => Ok(model.clone()),
        Some(Value::String(_)) => Err("Invalid request: 'model' must be a non-empty string"),
        Some(_) => Err("Invalid request: 'model' must be a string"),
        None => Err("Invalid request: missing required field 'model'"),
    }
}

// ============================================================================
// Response transformation helpers
// ============================================================================

/// Transform Ollama chat response to OpenAI chat completion format.
/// Takes processed content directly (already run through TrashCompactor::process_response_message).
fn extract_ollama_tool_calls(message: &Value) -> Option<Vec<Value>> {
    let tool_calls = message.get("tool_calls")?.as_array()?;

    let normalized = tool_calls
        .iter()
        .map(|tool_call| {
            let mut call = tool_call.clone();
            if let Some(obj) = call.as_object_mut() {
                obj.entry("type".to_string())
                    .or_insert_with(|| Value::String("function".to_string()));
            }
            call
        })
        .collect::<Vec<_>>();

    if normalized.is_empty() {
        None
    } else {
        Some(normalized)
    }
}

fn normalize_openai_tool_calls(tool_calls: &[Value]) -> Vec<Value> {
    tool_calls
        .iter()
        .enumerate()
        .filter_map(|(position, tool_call)| {
            let obj = tool_call.as_object()?;

            let index = obj
                .get("index")
                .and_then(|i| i.as_u64())
                .unwrap_or(position as u64);

            let tool_type = obj
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("function")
                .to_string();

            let function_name = obj
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();

            let arguments = match obj.get("function").and_then(|f| f.get("arguments")) {
                Some(Value::String(s)) => s.clone(),
                Some(other) => serde_json::to_string(other).unwrap_or_default(),
                None => String::new(),
            };

            let mut normalized = Map::new();
            normalized.insert("index".to_string(), Value::Number(index.into()));
            if let Some(id) = obj.get("id").and_then(|i| i.as_str()) {
                normalized.insert("id".to_string(), Value::String(id.to_string()));
            }
            normalized.insert("type".to_string(), Value::String(tool_type));
            normalized.insert(
                "function".to_string(),
                serde_json::json!({
                    "name": function_name,
                    "arguments": arguments
                }),
            );

            Some(Value::Object(normalized))
        })
        .collect()
}

fn ollama_tool_calls_to_anthropic_content(tool_calls: &[Value]) -> Vec<Value> {
    tool_calls
        .iter()
        .enumerate()
        .filter_map(|(idx, tc)| {
            let function = tc.get("function")?;
            let name = function.get("name").and_then(|n| n.as_str())?;

            let raw_input = function.get("arguments").cloned().unwrap_or(Value::Null);
            let parsed_input = match raw_input {
                Value::String(s) => serde_json::from_str::<Value>(&s)
                    .ok()
                    .filter(|v| v.is_object())
                    .unwrap_or_else(|| Value::Object(Map::new())),
                Value::Object(_) => raw_input,
                _ => Value::Object(Map::new()),
            };

            let id = tc
                .get("id")
                .and_then(|i| i.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| format!("toolu_{}", idx));

            Some(serde_json::json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": parsed_input
            }))
        })
        .collect()
}

fn ollama_to_openai_response(
    ollama_resp: &Value,
    model: &str,
    id: &str,
    created: u64,
    processed_content: &str,
    tool_calls: Option<&[Value]>,
) -> Value {
    let role = ollama_resp
        .get("message")
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        .unwrap_or("assistant");

    let done = ollama_resp
        .get("done")
        .and_then(|d| d.as_bool())
        .unwrap_or(true);

    let normalized_tool_calls = tool_calls.map(normalize_openai_tool_calls).unwrap_or_default();
    let has_tool_calls = !normalized_tool_calls.is_empty();
    let finish_reason = if has_tool_calls {
        "tool_calls"
    } else if done {
        "stop"
    } else {
        "length"
    };

    let mut message = serde_json::json!({
        "role": role,
        "content": processed_content
    });

    if has_tool_calls {
        message["tool_calls"] = Value::Array(normalized_tool_calls);
        if processed_content.is_empty() {
            message["content"] = Value::Null;
        }
    }

    // Preserve all top-level Ollama fields except transformed message.
    let mut response_obj = ollama_resp
        .as_object()
        .cloned()
        .unwrap_or_else(Map::new);
    response_obj.remove("message");

    // Ensure required OpenAI fields are present.
    response_obj.insert("id".to_string(), Value::String(id.to_string()));
    response_obj.insert(
        "object".to_string(),
        Value::String("chat.completion".to_string()),
    );
    response_obj.insert("created".to_string(), Value::Number(created.into()));
    response_obj.insert("model".to_string(), Value::String(model.to_string()));
    response_obj.insert(
        "choices".to_string(),
        serde_json::json!([{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason
        }]),
    );

    Value::Object(response_obj)
}

/// Transform Ollama chat response to Anthropic messages format.
/// Takes processed content directly (already run through TrashCompactor::process_response_message).
fn ollama_to_anthropic_response(
    ollama_resp: &Value,
    model: &str,
    id: &str,
    processed_content: &str,
    tool_calls: Option<&[Value]>,
) -> Value {
    let has_tool_calls = tool_calls.map(|tc| !tc.is_empty()).unwrap_or(false);
    let stop_reason = if has_tool_calls {
        "tool_use"
    } else {
        ollama_resp
            .get("done")
            .and_then(|d| d.as_bool())
            .map(|done| if done { "end_turn" } else { "max_tokens" })
            .unwrap_or("end_turn")
    };

    let mut content = vec![serde_json::json!({
        "type": "text",
        "text": processed_content
    })];

    if let Some(tc) = tool_calls {
        content.extend(ollama_tool_calls_to_anthropic_content(tc));
    }

    // Preserve all top-level Ollama fields except transformed message.
    let mut response_obj = ollama_resp
        .as_object()
        .cloned()
        .unwrap_or_else(Map::new);
    response_obj.remove("message");

    // Ensure required Anthropic fields are present.
    response_obj.insert("id".to_string(), Value::String(id.to_string()));
    response_obj.insert("type".to_string(), Value::String("message".to_string()));
    response_obj.insert("role".to_string(), Value::String("assistant".to_string()));
    response_obj.insert("model".to_string(), Value::String(model.to_string()));
    response_obj.insert("content".to_string(), Value::Array(content));
    response_obj.insert(
        "stop_reason".to_string(),
        Value::String(stop_reason.to_string()),
    );
    response_obj.insert("stop_sequence".to_string(), Value::Null);

    Value::Object(response_obj)
}

// ============================================================================
// Error response helper
// ============================================================================

/// Create an OpenAI-compatible error response.
fn error_response(status: StatusCode, message: &str) -> axum::response::Response<Body> {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "code": null
        }
    });
    (status, Json(body)).into_response()
}

// ============================================================================
// Main entry point
// ============================================================================

#[tokio::main]
async fn main() {
    if let Err(e) = run().await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

pub async fn run() -> anyhow::Result<()> {
    // Load configuration
    let config = AppConfig::load();

    println!("Ollama URL: {}", config.ollama_url);
    println!("Listen address: {}", config.listen_addr);
    if let Some(ref model) = config.compaction_model {
        println!("Compaction model: {}", model);
    } else {
        println!("Compaction: disabled (no compaction_model configured)");
    }

    // Create HTTP client with timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    // Create TrashCompactor (empty mappings = passthrough mode)
    let compactor = Arc::new(RwLock::new(trash_compactor::TrashCompactor::new()));

    // Create app state
    let state = AppState {
        client,
        ollama_url: config.ollama_url.clone(),
        compaction_model: config.compaction_model.clone(),
        default_max_context_size: config.default_max_context_size,
        model_overrides: config.model_overrides.clone(),
        compactor,
    };

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/models", get(models))
        .route("/v1/models/:model", get(get_model))
        .route("/v1/messages", post(messages))
        .route("/v1/responses", post(responses))
        .route("/v1/images/generations", post(image_generations))
        .with_state(state);

    println!("Starting server on http://{}", config.listen_addr);

    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// ============================================================================
// Compaction Functions
// ============================================================================

/// Get max context size for a model from config.
fn get_max_context_size(state: &AppState, model: &str) -> usize {
    state
        .model_overrides
        .get(model)
        .and_then(|o| o.max_context_size)
        .unwrap_or(state.default_max_context_size)
}

/// Call Ollama for compaction. Retry once on failure, then panic.
async fn call_compaction_model(
    client: &reqwest::Client,
    ollama_url: &str,
    compaction_model: &str,
    combined_text: &str,
) -> String {
    let user_content = format!("{}{}", trash_compactor::COMPACTION_USER_PREFIX, combined_text);

    for attempt in 0..2 {
        let request = serde_json::json!({
            "model": compaction_model,
            "messages": [
                {"role": "system", "content": trash_compactor::COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": &user_content}
            ],
            "stream": false
        });

        let result = client
            .post(&format!("{}/api/chat", ollama_url))
            .json(&request)
            .send()
            .await;

        match result {
            Ok(resp) if resp.status().is_success() => {
                match resp.json::<Value>().await {
                    Ok(body) => {
                        if let Some(content) = body
                            .get("message")
                            .and_then(|m| m.get("content"))
                            .and_then(|c| c.as_str())
                        {
                            return content.to_string();
                        }
                        // Malformed response — treat as failure
                        eprintln!(
                            "Compaction call failed (attempt {}): malformed response",
                            attempt + 1
                        );
                    }
                    Err(e) => {
                        eprintln!("Compaction call failed (attempt {}): failed to parse response: {}", attempt + 1, e);
                    }
                }
            }
            Ok(resp) => {
                eprintln!(
                    "Compaction call failed (attempt {}): HTTP {}",
                    attempt + 1,
                    resp.status()
                );
            }
            Err(e) => {
                eprintln!("Compaction call failed (attempt {}): {}", attempt + 1, e);
            }
        }

        if attempt == 0 {
            eprintln!("Retrying compaction call...");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    panic!("Compaction failed after 2 attempts — cannot proceed safely with over-budget context");
}

/// Run the full compaction workflow if needed.
/// This implements the multi-phase context compaction system.
async fn maybe_compact(
    state: &AppState,
    model: &str,
    messages: &[trash_compactor::Message],
) {
    // Get max context size for this model
    let max_context_size = get_max_context_size(state, model);

    // Check if compaction is enabled
    let Some(ref compaction_model) = state.compaction_model else {
        // Compaction disabled - just check and warn if over budget
        let compactor = state.compactor.read().await;
        let tokens = compactor.estimate_context_tokens(messages);
        if tokens > max_context_size {
            eprintln!(
                "Warning: Context over budget ({} > {} tokens) but compaction is disabled",
                tokens, max_context_size
            );
        }
        return;
    };

    // Reset compaction state for this request
    {
        let mut compactor = state.compactor.write().await;
        compactor.reset_compaction_state();
    }

    // Check initial token count
    let initial_tokens = {
        let compactor = state.compactor.read().await;
        compactor.estimate_context_tokens(messages)
    };

    if initial_tokens <= max_context_size {
        return; // Under budget, no compaction needed
    }

    println!(
        "Context over budget ({} > {} tokens), starting compaction",
        initial_tokens, max_context_size
    );

    // Phase 1: Skip low-priority messages
    let tokens_after_phase1 = {
        let mut compactor = state.compactor.write().await;
        compactor.skip_low_priority(messages)
    };

    if tokens_after_phase1 <= max_context_size {
        println!(
            "Phase 1 complete: context reduced to {} tokens",
            tokens_after_phase1
        );
        return;
    }

    // Phases 2-4: Progressive compaction
    loop {
        // Get the next compaction plan
        let plan = {
            let compactor = state.compactor.read().await;
            compactor.plan_compaction(messages, max_context_size)
        };

        let Some(plan) = plan else {
            // No more phases available
            let final_tokens = {
                let compactor = state.compactor.read().await;
                compactor.estimate_context_tokens(messages)
            };
            if final_tokens > max_context_size {
                eprintln!(
                    "Warning: Compaction complete but still over budget ({} > {} tokens)",
                    final_tokens, max_context_size
                );
            }
            break;
        };

        println!(
            "Running {:?} with {} messages",
            plan.phase,
            plan.messages_to_compact.len()
        );

        // Call compaction model (no lock held during this)
        let compacted_text = call_compaction_model(
            &state.client,
            &state.ollama_url,
            compaction_model,
            &plan.combined_text,
        )
        .await;

        // Determine survivor
        let survivor = if plan.survivor_index < messages.len() {
            messages[plan.survivor_index].clone()
        } else {
            plan.messages_to_compact.first().cloned().unwrap()
        };

        // Determine which messages to skip (all except survivor)
        let messages_to_skip: Vec<trash_compactor::Message> = plan
            .messages_to_compact
            .iter()
            .filter(|m| **m != survivor)
            .cloned()
            .collect();

        // Apply compaction result
        let result = trash_compactor::CompactionResult {
            phase: plan.phase,
            survivor,
            compacted_text,
            messages_to_skip,
            promote_to_high: plan.phase == trash_compactor::CompactionPhase::Phase3MediumAll,
        };

        {
            let mut compactor = state.compactor.write().await;
            compactor.apply_compaction(result);
        }

        // Check if we're now under budget
        let tokens_after = {
            let compactor = state.compactor.read().await;
            compactor.estimate_context_tokens(messages)
        };

        println!(
            "{:?} complete: context now {} tokens",
            plan.phase, tokens_after
        );

        if tokens_after <= max_context_size {
            break;
        }
    }
}

// ============================================================================
// Handlers
// ============================================================================

async fn chat_completions(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> axum::response::Response<Body> {
    println!("POST /v1/chat/completions");

    let model = match required_model_from_payload(&payload) {
        Ok(model) => model,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, msg),
    };

    // Extract stream flag
    let stream = payload
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    // Use current Unix timestamp for created
    let created: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let id = format!("chatcmpl-{}", uuid_timestamp());

    // Convert incoming messages to TrashCompactor format
    let incoming_messages = json_messages_to_compact(&payload);

    // Run through mappings first
    {
        let mut compactor = state.compactor.write().await;
        compactor.run_mappings(&model, incoming_messages.clone());
    }

    // Run context budget compaction if needed
    maybe_compact(&state, &model, &incoming_messages).await;

    // Run through compactor
    let compacted_messages: Vec<trash_compactor::Message> = {
        let compactor = state.compactor.read().await;
        compactor.compact(incoming_messages.clone()).collect()
    };

    // Convert back to JSON for Ollama
    let ollama_messages = compact_to_json_messages(compacted_messages);

    // Build Ollama request by starting from incoming payload and replacing messages only
    // This preserves all other fields exactly as provided.
    let mut ollama_request = payload.clone();
    if let Some(obj) = ollama_request.as_object_mut() {
        obj.insert("messages".to_string(), Value::Array(ollama_messages));
        // For streaming, ensure stream: true is explicitly set
        if stream {
            obj.insert("stream".to_string(), Value::Bool(true));
        }
    }

    // Send request to Ollama
    let ollama_endpoint = format!("{}/api/chat", state.ollama_url);

    let response = match state
        .client
        .post(&ollama_endpoint)
        .json(&ollama_request)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            let err_msg = format!("Ollama unavailable: {}", e);
            eprintln!("{}", err_msg);
            return error_response(StatusCode::BAD_GATEWAY, &err_msg);
        }
    };

    // Check for non-2xx status
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "(no body)".to_string());
        let err_msg = format!("Ollama error: {} {}", status, body);
        eprintln!("{}", err_msg);
        return error_response(StatusCode::BAD_GATEWAY, &err_msg);
    }

    if stream {
        // Streaming response: convert Ollama NDJSON to OpenAI SSE format
        handle_streaming_chat(response, model, id, created, state.compactor.clone()).await
    } else {
        // Non-streaming response
        let response_text = match response.text().await {
            Ok(text) => text,
            Err(e) => {
                let err_msg = format!("Failed to read Ollama response: {}", e);
                eprintln!("{}", err_msg);
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err_msg);
            }
        };

        let ollama_resp: Value = match serde_json::from_str(&response_text) {
            Ok(v) => v,
            Err(e) => {
                let err_msg = format!("Malformed Ollama response: {}", e);
                eprintln!("{} (response was: {})", err_msg, response_text);
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err_msg);
            }
        };

        // Validate response has expected fields
        if ollama_resp.get("message").is_none() {
            let err_msg = "Unexpected Ollama response format: missing 'message' field";
            eprintln!("{} (response was: {})", err_msg, response_text);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, err_msg);
        }

        // Extract content and role from Ollama response
        let raw_content = ollama_resp
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_string();

        let tool_calls = ollama_resp
            .get("message")
            .and_then(extract_ollama_tool_calls);

        let role = ollama_resp
            .get("message")
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or("assistant")
            .to_string();

        // Process the response message through TrashCompactor and strip priority markers
        let processed_content = {
            let mut compactor = state.compactor.write().await;
            let message = trash_compactor::Message {
                role,
                content: raw_content.clone(),
            };
            compactor.process_response_message(&message);
            trash_compactor::TrashCompactor::strip_priority_markers(&raw_content)
        };

        let openai_response = ollama_to_openai_response(
            &ollama_resp,
            &model,
            &id,
            created,
            &processed_content,
            tool_calls.as_deref(),
        );
        Json(openai_response).into_response()
    }
}

/// Handle streaming chat response from Ollama, converting to OpenAI SSE format.
/// Streams chunks incrementally, stripping priority markers from content as it passes through.
/// After stream completion, calls process_response_message for internal state tracking.
async fn handle_streaming_chat(
    response: reqwest::Response,
    model: String,
    id: String,
    created: u64,
    compactor: Arc<RwLock<trash_compactor::TrashCompactor>>,
) -> axum::response::Response<Body> {
    use tokio_stream::wrappers::ReceiverStream;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    // Spawn a task to process the Ollama stream
    tokio::spawn(async move {
        let mut byte_buffer = Vec::new();
        let mut stream = response.bytes_stream();

        // Buffer to accumulate all content for post-stream processing
        let mut accumulated_content = String::new();
        let mut accumulated_tool_calls: Vec<Value> = Vec::new();

        // Track if we've emitted the role in the first chunk
        let mut emitted_role = false;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(bytes) => {
                    byte_buffer.extend_from_slice(&bytes);

                    // Process complete lines
                    while let Some(newline_pos) = byte_buffer.iter().position(|&b| b == b'\n') {
                        let line_bytes: Vec<u8> = byte_buffer.drain(..=newline_pos).collect();
                        let line = String::from_utf8_lossy(&line_bytes);
                        let line = line.trim();

                        if line.is_empty() {
                            continue;
                        }

                        // Parse Ollama JSON chunk
                        match serde_json::from_str::<Value>(line) {
                            Ok(ollama_chunk) => {
                                // Extract and accumulate tool calls
                                if let Some(chunk_tool_calls) = ollama_chunk
                                    .get("message")
                                    .and_then(extract_ollama_tool_calls)
                                {
                                    // Merge tool call deltas - for streaming, Ollama sends complete tool_calls on done
                                    if !chunk_tool_calls.is_empty() {
                                        accumulated_tool_calls = chunk_tool_calls;
                                    }
                                }

                                // Extract content from chunk
                                let raw_content = ollama_chunk
                                    .get("message")
                                    .and_then(|m| m.get("content"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("");

                                // Accumulate raw content for post-stream processing
                                accumulated_content.push_str(raw_content);

                                let done = ollama_chunk
                                    .get("done")
                                    .and_then(|d| d.as_bool())
                                    .unwrap_or(false);

                                // Strip priority markers from content for emission
                                let stripped_content = trash_compactor::TrashCompactor::strip_priority_markers(raw_content);

                                // Build the delta for this chunk
                                let mut delta = serde_json::json!({});

                                // Include role only in the first chunk
                                if !emitted_role {
                                    delta["role"] = Value::String("assistant".to_string());
                                    emitted_role = true;
                                }

                                // Include content if there's any (after stripping)
                                if !stripped_content.is_empty() {
                                    delta["content"] = Value::String(stripped_content);
                                }

                                // Determine finish_reason
                                let normalized_tool_calls = if done {
                                    Some(normalize_openai_tool_calls(&accumulated_tool_calls))
                                } else {
                                    None
                                };

                                let has_tool_calls = normalized_tool_calls
                                    .as_ref()
                                    .map(|tc| !tc.is_empty())
                                    .unwrap_or(false);

                                let finish_reason = if done {
                                    if has_tool_calls {
                                        Some("tool_calls")
                                    } else {
                                        Some("stop")
                                    }
                                } else {
                                    None
                                };

                                // Include tool_calls on the final chunk if present
                                if done && has_tool_calls {
                                    delta["tool_calls"] = Value::Array(
                                        normalized_tool_calls.unwrap_or_default(),
                                    );
                                    if raw_content.is_empty() {
                                        delta["content"] = Value::Null;
                                    }
                                }

                                // Build the SSE chunk
                                let sse_chunk = serde_json::json!({
                                    "id": &id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": &model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": delta,
                                        "finish_reason": finish_reason
                                    }]
                                });

                                match serde_json::to_string(&sse_chunk) {
                                    Ok(json) => {
                                        if tx.send(Ok(Event::default().data(json))).await.is_err() {
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("Failed to serialize streaming chunk: {}", e);
                                    }
                                }

                                // When done, send [DONE] marker and process accumulated content
                                if done {
                                    // Send [DONE]
                                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;

                                    // Process the complete accumulated content through TrashCompactor
                                    // This updates internal state but doesn't affect already-streamed output
                                    let mut compactor_guard = compactor.write().await;
                                    compactor_guard.process_response_message(&trash_compactor::Message {
                                        role: "assistant".to_string(),
                                        content: accumulated_content.clone(),
                                    });

                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!(
                                    "Failed to parse Ollama stream chunk: {} (line: {})",
                                    e, line
                                );
                            }
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Stream error: {}", e);
                    break;
                }
            }
        }

        // If we never emitted a role but had content, send fallback
        if !emitted_role && !accumulated_content.is_empty() {
            // Process the complete accumulated content through TrashCompactor
            let stripped_content = trash_compactor::TrashCompactor::strip_priority_markers(&accumulated_content);
            
            let mut compactor_guard = compactor.write().await;
            compactor_guard.process_response_message(&trash_compactor::Message {
                role: "assistant".to_string(),
                content: accumulated_content.clone(),
            });
            drop(compactor_guard);

            let normalized_tool_calls = normalize_openai_tool_calls(&accumulated_tool_calls);
            let has_tool_calls = !normalized_tool_calls.is_empty();

            // Emit single chunk with role and content
            let mut delta = serde_json::json!({
                "role": "assistant",
                "content": stripped_content
            });

            if has_tool_calls {
                delta["tool_calls"] = Value::Array(normalized_tool_calls);
            }

            let sse_chunk = serde_json::json!({
                "id": &id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": delta,
                    "finish_reason": if has_tool_calls { "tool_calls" } else { "stop" }
                }]
            });

            if let Ok(json) = serde_json::to_string(&sse_chunk) {
                let _ = tx.send(Ok(Event::default().data(json))).await;
            }

            let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
        }
    });

    let event_stream = ReceiverStream::new(rx);
    Sse::new(event_stream).into_response()
}

/// Generate a simple timestamp-based ID.
fn uuid_timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{:x}", now)
}

// ============================================================================
// Anthropic-style /v1/messages handler with Ollama passthrough
// ============================================================================

async fn messages(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> axum::response::Response<Body> {
    println!("POST /v1/messages");

    // Anthropic-style doesn't support streaming in this implementation
    let stream = payload
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    if stream {
        return error_response(
            StatusCode::NOT_IMPLEMENTED,
            "Streaming not yet implemented for /v1/messages endpoint",
        );
    }

    let model = match required_model_from_payload(&payload) {
        Ok(model) => model,
        Err(msg) => return error_response(StatusCode::BAD_REQUEST, msg),
    };

    let id = format!("msg_{}", uuid_timestamp());

    // Convert Anthropic-format messages to TrashCompactor format
    let mut messages: Vec<trash_compactor::Message> = Vec::new();

    // Handle system prompt
    if let Some(system) = payload.get("system").and_then(|s| s.as_str()) {
        messages.push(trash_compactor::Message {
            role: "system".to_string(),
            content: system.to_string(),
        });
    }

    // Handle messages array
    if let Some(msgs) = payload.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg
                .get("role")
                .and_then(|r| r.as_str())
                .unwrap_or("user")
                .to_string();

            let content = match msg.get("content") {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Array(arr)) => arr
                    .iter()
                    .filter_map(|block| {
                        block
                            .get("type")
                            .and_then(|t| t.as_str())
                            .filter(|&t| t == "text")
                            .and_then(|_| block.get("text"))
                            .and_then(|t| t.as_str())
                    })
                    .collect::<Vec<_>>()
                    .join(" "),
                _ => String::new(),
            };

            messages.push(trash_compactor::Message { role, content });
        }
    }

    // Run through mappings first
    {
        let mut compactor = state.compactor.write().await;
        compactor.run_mappings(&model, messages.clone());
    }

    // Run context budget compaction if needed
    maybe_compact(&state, &model, &messages).await;

    // Run through compactor
    let compacted_messages: Vec<trash_compactor::Message> = {
        let compactor = state.compactor.read().await;
        compactor.compact(messages.clone()).collect()
    };

    // Convert to Ollama format
    let ollama_messages = compact_to_json_messages(compacted_messages);

    // Build Ollama request by starting from incoming payload and replacing messages only.
    // Do not mutate/remove other Anthropic fields from the forwarded payload blob.
    let mut ollama_request = payload.clone();
    if let Some(obj) = ollama_request.as_object_mut() {
        obj.insert("messages".to_string(), Value::Array(ollama_messages));
    }

    // Send request to Ollama
    let ollama_endpoint = format!("{}/api/chat", state.ollama_url);

    let response = match state
        .client
        .post(&ollama_endpoint)
        .json(&ollama_request)
        .send()
        .await
    {
        Ok(resp) => resp,
        Err(e) => {
            let err_msg = format!("Ollama unavailable: {}", e);
            eprintln!("{}", err_msg);
            return error_response(StatusCode::BAD_GATEWAY, &err_msg);
        }
    };

    // Check for non-2xx status
    if !response.status().is_success() {
        let status = response.status();
        let body = response
            .text()
            .await
            .unwrap_or_else(|_| "(no body)".to_string());
        let err_msg = format!("Ollama error: {} {}", status, body);
        eprintln!("{}", err_msg);
        return error_response(StatusCode::BAD_GATEWAY, &err_msg);
    }

    // Parse response
    let response_text = match response.text().await {
        Ok(text) => text,
        Err(e) => {
            let err_msg = format!("Failed to read Ollama response: {}", e);
            eprintln!("{}", err_msg);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err_msg);
        }
    };

    let ollama_resp: Value = match serde_json::from_str(&response_text) {
        Ok(v) => v,
        Err(e) => {
            let err_msg = format!("Malformed Ollama response: {}", e);
            eprintln!("{} (response was: {})", err_msg, response_text);
            return error_response(StatusCode::INTERNAL_SERVER_ERROR, &err_msg);
        }
    };

    // Validate response has expected fields
    if ollama_resp.get("message").is_none() {
        let err_msg = "Unexpected Ollama response format: missing 'message' field";
        eprintln!("{} (response was: {})", err_msg, response_text);
        return error_response(StatusCode::INTERNAL_SERVER_ERROR, err_msg);
    }

    // Extract content and role from Ollama response
    let raw_content = ollama_resp
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let tool_calls = ollama_resp
        .get("message")
        .and_then(extract_ollama_tool_calls);

    let role = ollama_resp
        .get("message")
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        .unwrap_or("assistant")
        .to_string();

    // Process the response message through TrashCompactor and strip priority markers
    let processed_content = {
        let mut compactor = state.compactor.write().await;
        let message = trash_compactor::Message {
            role,
            content: raw_content.clone(),
        };
        compactor.process_response_message(&message);
        trash_compactor::TrashCompactor::strip_priority_markers(&raw_content)
    };

    let anthropic_response = ollama_to_anthropic_response(
        &ollama_resp,
        &model,
        &id,
        &processed_content,
        tool_calls.as_deref(),
    );
    Json(anthropic_response).into_response()
}

// ============================================================================
// Stub handlers (unchanged)
// ============================================================================

async fn completions(Json(payload): Json<Value>) -> Json<Value> {
    println!("POST /v1/completions payload: {:#}", payload);

    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("stub-model")
        .to_string();

    Json(serde_json::json!({
        "id": "cmpl-stub",
        "object": "text_completion",
        "created": 1234567890,
        "model": model,
        "choices": [{
            "text": "This is a stub response.",
            "index": 0,
            "finish_reason": "stop"
        }]
    }))
}

async fn embeddings(Json(payload): Json<Value>) -> Json<Value> {
    println!("POST /v1/embeddings payload: {:#}", payload);

    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("stub-model")
        .to_string();

    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": vec![0.1; 384],
            "index": 0
        }],
        "model": model
    }))
}

async fn models() -> Json<Value> {
    println!("GET /v1/models");
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": "qwen3-coder:30b",
            "object": "model",
            "created": 0,
            "owned_by": "trash-compactor"
        }]
    }))
}

async fn get_model(Path(model_id): Path<String>) -> Json<Value> {
    println!("GET /v1/models/{}", model_id);
    Json(serde_json::json!({
        "id": model_id,
        "object": "model",
        "created": 0,
        "owned_by": "trash-compactor",
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": null,
            "families": null
        }
    }))
}

async fn responses(Json(payload): Json<Value>) -> Json<Value> {
    println!("POST /v1/responses payload: {:#}", payload);

    let model = payload
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("stub-model")
        .to_string();

    Json(serde_json::json!({
        "id": "resp.stub",
        "object": "response",
        "created": 1234567890,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello! This is a stub response."
            },
            "finish_reason": "stop"
        }]
    }))
}

async fn image_generations(Json(payload): Json<Value>) -> Json<Value> {
    println!("POST /v1/images/generations payload: {:#}", payload);
    Json(serde_json::json!({
        "created": 1234567890,
        "data": [{
            "b64_json": null,
            "url": "https://example.com/image.png",
            "revised_prompt": "A stub image"
        }]
    }))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test: OpenAI non-stream tool_calls passthrough and output metadata preservation.
    /// This test verifies that:
    /// 1. tool_calls from Ollama response are correctly passed through to OpenAI format
    /// 2. Additional output metadata fields from Ollama are preserved in the response
    #[test]
    fn test_openai_non_stream_tool_calls_passthrough() {
        // Simulate an Ollama response with tool_calls and extra metadata
        let ollama_resp = serde_json::json!({
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"San Francisco\"}"
                        }
                    }
                ]
            },
            "done": true,
            "total_duration": 1234567890,
            "load_duration": 1000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 5000000,
            "eval_count": 20,
            "eval_duration": 10000000,
            "custom_metadata": "should_be_preserved"
        });

        // Extract tool_calls using the helper function
        let tool_calls = ollama_resp
            .get("message")
            .and_then(extract_ollama_tool_calls);

        // Transform to OpenAI format
        let openai_response = ollama_to_openai_response(
            &ollama_resp,
            "test-model",
            "chatcmpl-test",
            1234567890,
            "", // processed_content is empty when tool_calls present
            tool_calls.as_deref(),
        );

        // Verify tool_calls are present in the response
        let choices = openai_response.get("choices").and_then(|c| c.as_array()).unwrap();
        assert_eq!(choices.len(), 1);
        
        let message = &choices[0]["message"];
        let response_tool_calls = message.get("tool_calls").and_then(|tc| tc.as_array()).unwrap();
        assert_eq!(response_tool_calls.len(), 1);
        
        // Verify tool_call structure
        let tool_call = &response_tool_calls[0];
        assert_eq!(tool_call.get("index").and_then(|i| i.as_u64()), Some(0));
        assert_eq!(tool_call.get("id").and_then(|i| i.as_str()), Some("call_123"));
        assert_eq!(tool_call.get("type").and_then(|t| t.as_str()), Some("function"));
        
        let function = tool_call.get("function").unwrap();
        assert_eq!(function.get("name").and_then(|n| n.as_str()), Some("get_weather"));
        assert_eq!(
            function.get("arguments").and_then(|a| a.as_str()),
            Some("{\"location\": \"San Francisco\"}")
        );

        // Verify finish_reason is "tool_calls"
        assert_eq!(
            choices[0].get("finish_reason").and_then(|f| f.as_str()),
            Some("tool_calls")
        );

        // Verify content is null when empty with tool_calls
        assert!(message.get("content").unwrap().is_null());

        // Verify extra metadata is preserved
        assert_eq!(
            openai_response.get("custom_metadata").and_then(|m| m.as_str()),
            Some("should_be_preserved")
        );
    }

    #[test]
    fn test_normalize_openai_tool_calls_for_streaming_delta() {
        let tool_calls = vec![
            serde_json::json!({
                "id": "call_1",
                "function": {
                    "name": "lookup",
                    "arguments": {"location": "San Francisco"}
                }
            }),
            serde_json::json!({
                "index": 7,
                "type": "function",
                "function": {
                    "name": "sum",
                    "arguments": [1, 2, 3]
                }
            }),
        ];

        let normalized = normalize_openai_tool_calls(&tool_calls);
        assert_eq!(normalized.len(), 2);

        assert_eq!(normalized[0].get("index").and_then(|i| i.as_u64()), Some(0));
        assert_eq!(normalized[0].get("type").and_then(|t| t.as_str()), Some("function"));
        assert_eq!(
            normalized[0]
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str()),
            Some("{\"location\":\"San Francisco\"}")
        );

        assert_eq!(normalized[1].get("index").and_then(|i| i.as_u64()), Some(7));
        assert_eq!(
            normalized[1]
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str()),
            Some("[1,2,3]")
        );
    }

    /// Test that input passthrough preserves extra fields in chat completions request.
    #[test]
    fn test_chat_completions_input_passthrough() {
        // Simulate incoming payload with extra fields
        let payload = serde_json::json!({
            "model": "test-model",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "tools": [{"type": "function", "function": {"name": "test"}}],
            "tool_choice": "auto",
            "custom_field": "should_be_preserved"
        });

        // Extract model
        let model = required_model_from_payload(&payload).unwrap();
        assert_eq!(model, "test-model");

        // Simulate the passthrough logic (what happens in chat_completions handler)
        // Convert messages
        let incoming_messages = json_messages_to_compact(&payload);
        let compacted_messages: Vec<trash_compactor::Message> = incoming_messages;
        let ollama_messages = compact_to_json_messages(compacted_messages);

        // Build Ollama request by starting from incoming payload
        let mut ollama_request = payload.clone();
        if let Some(obj) = ollama_request.as_object_mut() {
            obj.insert("messages".to_string(), Value::Array(ollama_messages));
        }

        // Verify extra fields are preserved
        assert_eq!(
            ollama_request.get("temperature").and_then(|t| t.as_f64()),
            Some(0.7)
        );
        assert_eq!(
            ollama_request.get("max_tokens").and_then(|t| t.as_u64()),
            Some(100)
        );
        assert_eq!(
            ollama_request.get("top_p").and_then(|t| t.as_f64()),
            Some(0.9)
        );
        assert!(ollama_request.get("tools").is_some());
        assert_eq!(
            ollama_request.get("tool_choice").and_then(|t| t.as_str()),
            Some("auto")
        );
        assert_eq!(
            ollama_request.get("custom_field").and_then(|f| f.as_str()),
            Some("should_be_preserved")
        );

        // Verify model is preserved
        assert_eq!(
            ollama_request.get("model").and_then(|m| m.as_str()),
            Some("test-model")
        );

        // Verify stream key remains untouched (not force-added)
        assert!(ollama_request.get("stream").is_none());
    }

    /// Test that /v1/messages input passthrough only replaces messages.
    #[test]
    fn test_messages_input_passthrough_only_replaces_messages() {
        let payload = serde_json::json!({
            "model": "anthropic-model",
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
            ],
            "metadata": {"tenant": "abc"},
            "custom": 42
        });

        // Simulate conversion and passthrough logic from /v1/messages handler.
        let mut compact_messages = vec![trash_compactor::Message {
            role: "system".to_string(),
            content: payload
                .get("system")
                .and_then(|s| s.as_str())
                .unwrap_or_default()
                .to_string(),
        }];

        if let Some(msgs) = payload.get("messages").and_then(|m| m.as_array()) {
            for msg in msgs {
                let role = msg
                    .get("role")
                    .and_then(|r| r.as_str())
                    .unwrap_or("user")
                    .to_string();
                let content = msg
                    .get("content")
                    .and_then(|c| c.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|block| {
                                block
                                    .get("type")
                                    .and_then(|t| t.as_str())
                                    .filter(|&t| t == "text")
                                    .and_then(|_| block.get("text"))
                                    .and_then(|t| t.as_str())
                            })
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                compact_messages.push(trash_compactor::Message { role, content });
            }
        }

        let ollama_messages = compact_to_json_messages(compact_messages);
        let mut ollama_request = payload.clone();
        if let Some(obj) = ollama_request.as_object_mut() {
            obj.insert("messages".to_string(), Value::Array(ollama_messages));
        }

        // Verify all non-messages fields are untouched.
        assert_eq!(
            ollama_request.get("system").and_then(|s| s.as_str()),
            Some("You are helpful")
        );
        assert_eq!(
            ollama_request
                .get("metadata")
                .and_then(|m| m.get("tenant"))
                .and_then(|t| t.as_str()),
            Some("abc")
        );
        assert_eq!(ollama_request.get("custom").and_then(|c| c.as_i64()), Some(42));
        assert!(ollama_request.get("stream").is_none());
    }

    /// Test that output passthrough preserves extra fields from Ollama response.
    #[test]
    fn test_output_passthrough_preserves_metadata() {
        // Simulate Ollama response with extra metadata
        let ollama_resp = serde_json::json!({
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello, world!"
            },
            "done": true,
            "total_duration": 1234567890,
            "load_duration": 1000000,
            "prompt_eval_count": 10,
            "prompt_eval_duration": 5000000,
            "eval_count": 20,
            "eval_duration": 10000000,
            "extra_field": "preserved_value",
            "another_field": {"nested": "data"}
        });

        let openai_response = ollama_to_openai_response(
            &ollama_resp,
            "test-model",
            "chatcmpl-test",
            1234567890,
            "Hello, world!",
            None,
        );

        // Verify extra fields are preserved
        assert_eq!(
            openai_response.get("extra_field").and_then(|f| f.as_str()),
            Some("preserved_value")
        );
        assert_eq!(
            openai_response.get("another_field")
                .and_then(|f| f.get("nested").and_then(|n| n.as_str())),
            Some("data")
        );
        assert_eq!(
            openai_response.get("created_at").and_then(|c| c.as_str()),
            Some("2024-01-01T00:00:00Z")
        );
        assert_eq!(openai_response.get("done").and_then(|d| d.as_bool()), Some(true));
        assert_eq!(
            openai_response
                .get("total_duration")
                .and_then(|d| d.as_u64()),
            Some(1234567890)
        );

        // Verify standard fields are set correctly
        assert_eq!(openai_response.get("id").and_then(|i| i.as_str()), Some("chatcmpl-test"));
        assert_eq!(openai_response.get("object").and_then(|o| o.as_str()), Some("chat.completion"));
        assert_eq!(openai_response.get("model").and_then(|m| m.as_str()), Some("test-model"));
    }

    /// Test Anthropic output passthrough preserves extra fields.
    #[test]
    fn test_anthropic_output_passthrough_preserves_metadata() {
        // Simulate Ollama response with extra metadata
        let ollama_resp = serde_json::json!({
            "model": "test-model",
            "created_at": "2024-01-01T00:00:00Z",
            "message": {
                "role": "assistant",
                "content": "Hello!"
            },
            "done": true,
            "custom_anthropic_field": "preserved"
        });

        let anthropic_response = ollama_to_anthropic_response(
            &ollama_resp,
            "test-model",
            "msg_test",
            "Hello!",
            None,
        );

        // Verify extra field is preserved
        assert_eq!(
            anthropic_response.get("custom_anthropic_field").and_then(|f| f.as_str()),
            Some("preserved")
        );
        assert_eq!(
            anthropic_response.get("created_at").and_then(|c| c.as_str()),
            Some("2024-01-01T00:00:00Z")
        );
        assert_eq!(anthropic_response.get("done").and_then(|d| d.as_bool()), Some(true));

        // Verify standard Anthropic fields are set correctly
        assert_eq!(anthropic_response.get("id").and_then(|i| i.as_str()), Some("msg_test"));
        assert_eq!(anthropic_response.get("type").and_then(|t| t.as_str()), Some("message"));
        assert_eq!(anthropic_response.get("role").and_then(|r| r.as_str()), Some("assistant"));
        assert_eq!(anthropic_response.get("stop_reason").and_then(|s| s.as_str()), Some("end_turn"));
    }

    /// Test tool_calls extraction from Ollama response.
    #[test]
    fn test_extract_ollama_tool_calls() {
        // Test with valid tool_calls
        let message_with_tools = serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "search",
                        "arguments": "{\"query\": \"test\"}"
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": "{\"expr\": \"1+1\"}"
                    }
                }
            ]
        });

        let tool_calls = extract_ollama_tool_calls(&message_with_tools).unwrap();
        assert_eq!(tool_calls.len(), 2);

        // Verify type field is added when missing
        assert_eq!(tool_calls[0].get("type").and_then(|t| t.as_str()), Some("function"));
        assert_eq!(tool_calls[1].get("type").and_then(|t| t.as_str()), Some("function"));

        // Test with no tool_calls
        let message_without_tools = serde_json::json!({
            "role": "assistant",
            "content": "Hello"
        });
        assert!(extract_ollama_tool_calls(&message_without_tools).is_none());

        // Test with empty tool_calls array
        let message_empty_tools = serde_json::json!({
            "role": "assistant",
            "content": "",
            "tool_calls": []
        });
        assert!(extract_ollama_tool_calls(&message_empty_tools).is_none());
    }

    /// Test model validation from payload.
    #[test]
    fn test_required_model_from_payload() {
        // Valid model
        let valid_payload = serde_json::json!({"model": "test-model"});
        assert_eq!(required_model_from_payload(&valid_payload), Ok("test-model".to_string()));

        // Empty model string
        let empty_model = serde_json::json!({"model": "   "});
        assert!(required_model_from_payload(&empty_model).is_err());

        // Missing model
        let missing_model = serde_json::json!({"messages": []});
        assert!(required_model_from_payload(&missing_model).is_err());

        // Non-string model
        let non_string_model = serde_json::json!({"model": 123});
        assert!(required_model_from_payload(&non_string_model).is_err());
    }
}
