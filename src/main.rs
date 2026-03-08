use axum::{
    body::Body,
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use serde_json::Value;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

// ============================================================================
// App State
// ============================================================================

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    ollama_url: String,
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
fn ollama_to_openai_response(
    ollama_resp: &Value,
    model: &str,
    id: &str,
    created: u64,
    processed_content: &str,
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

    let finish_reason = if done { "stop" } else { "length" };

    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": role,
                "content": processed_content
            },
            "finish_reason": finish_reason
        }]
    })
}

/// Transform Ollama chat response to Anthropic messages format.
/// Takes processed content directly (already run through TrashCompactor::process_response_message).
fn ollama_to_anthropic_response(
    ollama_resp: &Value,
    model: &str,
    id: &str,
    processed_content: &str,
) -> Value {
    let stop_reason = ollama_resp
        .get("done")
        .and_then(|d| d.as_bool())
        .map(|done| if done { "end_turn" } else { "max_tokens" })
        .unwrap_or("end_turn");

    serde_json::json!({
        "id": id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{
            "type": "text",
            "text": processed_content
        }],
        "stop_reason": stop_reason,
        "stop_sequence": null
    })
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
    // Load configuration from environment
    let ollama_url =
        std::env::var("OLLAMA_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434".to_string());

    println!("Ollama URL: {}", ollama_url);

    // Create HTTP client with timeout
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;

    // Create TrashCompactor (empty mappings = passthrough mode)
    let compactor = Arc::new(RwLock::new(trash_compactor::TrashCompactor::new()));

    // Create app state
    let state = AppState {
        client,
        ollama_url,
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

    let addr = "127.0.0.1:11435";
    println!("Starting server on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
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
        compactor.run_mappings(incoming_messages.clone());
    }

    // Run through compactor
    let compacted_messages: Vec<trash_compactor::Message> = {
        let compactor = state.compactor.read().await;
        compactor.compact(incoming_messages).collect()
    };

    // Convert back to JSON for Ollama
    let ollama_messages = compact_to_json_messages(compacted_messages);

    // Build Ollama request
    let ollama_request = serde_json::json!({
        "model": model,
        "messages": ollama_messages,
        "stream": stream
    });

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

        let role = ollama_resp
            .get("message")
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            .unwrap_or("assistant")
            .to_string();

        // Process the response message through TrashCompactor
        let processed_message = {
            let mut compactor = state.compactor.write().await;
            compactor.process_response_message(trash_compactor::Message {
                role,
                content: raw_content,
            })
        };

        let openai_response = ollama_to_openai_response(
            &ollama_resp,
            &model,
            &id,
            created,
            dbg!(&processed_message.content),
        );
        Json(openai_response).into_response()
    }
}

/// Handle streaming chat response from Ollama, converting to OpenAI SSE format.
/// Buffers content and processes through TrashCompactor before emitting.
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

        // Buffer to accumulate all content for processing
        let mut accumulated_content = String::new();

        // Track if we've received any content
        let mut has_content = false;

        // Track whether we've already emitted the final assistant response
        let mut emitted_final_response = false;

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
                                let content = ollama_chunk
                                    .get("message")
                                    .and_then(|m| m.get("content"))
                                    .and_then(|c| c.as_str())
                                    .unwrap_or("");

                                let done = ollama_chunk
                                    .get("done")
                                    .and_then(|d| d.as_bool())
                                    .unwrap_or(false);

                                // Accumulate content for later processing
                                accumulated_content.push_str(content);
                                if !content.is_empty() {
                                    has_content = true;
                                }

                                // When done, process the accumulated content and emit
                                if done {
                                    // Process the complete response message through TrashCompactor
                                    let processed_content = {
                                        let mut compactor_guard = compactor.write().await;
                                        let processed = compactor_guard.process_response_message(
                                            trash_compactor::Message {
                                                role: "assistant".to_string(),
                                                content: accumulated_content.clone(),
                                            },
                                        );
                                        processed.content
                                    };

                                    // Emit first chunk with role and processed content
                                    let first_chunk = serde_json::json!({
                                        "id": &id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": &model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {
                                                "role": "assistant",
                                                "content": processed_content
                                            },
                                            "finish_reason": null
                                        }]
                                    });

                                    match serde_json::to_string(&first_chunk) {
                                        Ok(json) => {
                                            if tx
                                                .send(Ok(Event::default().data(json)))
                                                .await
                                                .is_err()
                                            {
                                                break;
                                            }
                                        }
                                        Err(e) => {
                                            eprintln!("Failed to serialize streaming chunk: {}", e);
                                        }
                                    }

                                    // Emit final chunk with finish_reason
                                    let final_chunk = serde_json::json!({
                                        "id": &id,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": &model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": "stop"
                                        }]
                                    });

                                    match serde_json::to_string(&final_chunk) {
                                        Ok(json) => {
                                            let _ = tx.send(Ok(Event::default().data(json))).await;
                                        }
                                        Err(e) => {
                                            eprintln!("Failed to serialize final chunk: {}", e);
                                        }
                                    }

                                    // Send [DONE]
                                    let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
                                    emitted_final_response = true;
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

        // If we never sent a done chunk but had content, send processed content
        if has_content && !emitted_final_response {
            // Process the complete response message through TrashCompactor
            let processed_content = {
                let mut compactor_guard = compactor.write().await;
                let processed =
                    compactor_guard.process_response_message(trash_compactor::Message {
                        role: "assistant".to_string(),
                        content: accumulated_content.clone(),
                    });
                processed.content
            };

            // Emit first chunk with role and processed content
            let first_chunk = serde_json::json!({
                "id": &id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": processed_content
                    },
                    "finish_reason": null
                }]
            });

            if let Ok(json) = serde_json::to_string(&first_chunk) {
                let _ = tx.send(Ok(Event::default().data(json))).await;
            }

            // Emit final chunk with finish_reason
            let final_chunk = serde_json::json!({
                "id": &id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": &model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            });

            if let Ok(json) = serde_json::to_string(&final_chunk) {
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
        compactor.run_mappings(messages.clone());
    }

    // Run through compactor
    let compacted_messages: Vec<trash_compactor::Message> = {
        let compactor = state.compactor.read().await;
        compactor.compact(messages).collect()
    };

    // Convert to Ollama format
    let ollama_messages = compact_to_json_messages(compacted_messages);

    // Build Ollama request
    let ollama_request = serde_json::json!({
        "model": model,
        "messages": ollama_messages,
        "stream": false
    });

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

    let role = ollama_resp
        .get("message")
        .and_then(|m| m.get("role"))
        .and_then(|r| r.as_str())
        .unwrap_or("assistant")
        .to_string();

    // Process the response message through TrashCompactor
    let processed_message = {
        let mut compactor = state.compactor.write().await;
        compactor.process_response_message(trash_compactor::Message {
            role,
            content: raw_content,
        })
    };

    let anthropic_response =
        ollama_to_anthropic_response(&ollama_resp, &model, &id, &processed_message.content);
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
