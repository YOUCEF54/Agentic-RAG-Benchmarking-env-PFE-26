use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: AssistantMessage,
}

#[derive(Deserialize)]
struct AssistantMessage {
    content: String,
}

pub async fn ask(client: &Client, api_key: &str, model: &str, context: &str, question: &str) -> String {
    let prompt = format!(
        "Use the following context to answer the question.\n\nContext:\n{}\n\nQuestion: {}",
        context, question
    );

    let body = ChatRequest {
        model: model.to_string(),
        messages: vec![Message {
            role: "user".to_string(),
            content: prompt,
        }],
    };

    let res = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .unwrap()
        .json::<ChatResponse>()
        .await
        .unwrap();

    res.choices[0].message.content.clone()
}