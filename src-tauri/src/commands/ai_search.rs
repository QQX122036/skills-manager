use std::sync::Arc;
use tauri::State;

use crate::core::{
    ai_search_api,
    error::AppError,
    skill_store::SkillStore,
    skillssh_api::SkillsShSkill,
};

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize)]
pub struct AiSearchResult {
    pub thinking: String,
    pub skills: Vec<SkillsShSkill>,
}

#[derive(Debug, Serialize)]
pub struct AiSkillAnalysisResult {
    pub skillId: String,
    pub skillName: String,
    pub source: String,
    pub score: f64,
    pub description: String,
    pub howToUse: String,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct DeepSearchResult {
    pub thinking: String,
    pub totalFound: usize,
    pub analyzed: Vec<AiSkillAnalysisResult>,
    pub searchStrategy: String,
    pub channelsUsed: Vec<String>,
    pub verificationPassed: usize,
}

/// Conversation turn for multi-round search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String, // "user" or "assistant"
    pub content: String,
    pub timestamp: i64,
}

#[tauri::command]
pub async fn search_with_ai_api(
    query: String,
    store: State<'_, Arc<SkillStore>>,
) -> Result<AiSearchResult, AppError> {
    let ai_api_url = store
        .get_setting("ai_api_url")
        .map_err(AppError::db)?
        .filter(|v| !v.is_empty())
        .unwrap_or_default();
    let ai_api_key = store
        .get_setting("ai_api_key")
        .map_err(AppError::db)?
        .filter(|k| !k.is_empty())
        .ok_or_else(|| AppError::network(anyhow::anyhow!("AI API key not configured")))?;
    let proxy_url = store.proxy_url();

    let proxy = proxy_url;
    let (thinking, skills) = tauri::async_runtime::spawn_blocking(move || {
        ai_search_api::search(&ai_api_url, &ai_api_key, &query, proxy.as_deref())
            .map_err(AppError::network)
    })
    .await??;

    Ok(AiSearchResult { thinking, skills })
}

#[tauri::command]
pub async fn deep_search_with_ai(
    query: String,
    store: State<'_, Arc<SkillStore>>,
) -> Result<DeepSearchResult, AppError> {
    let ai_api_url = store
        .get_setting("ai_api_url")
        .map_err(AppError::db)?
        .filter(|v| !v.is_empty())
        .unwrap_or_default();
    let ai_api_key = store
        .get_setting("ai_api_key")
        .map_err(AppError::db)?
        .filter(|k| !k.is_empty())
        .ok_or_else(|| AppError::network(anyhow::anyhow!("AI API key not configured")))?;
    let proxy_url = store.proxy_url();

    let proxy = proxy_url;
    let result = tauri::async_runtime::spawn_blocking(move || {
        ai_search_api::deep_search(&ai_api_url, &ai_api_key, &query, proxy.as_deref())
            .map_err(AppError::network)
    })
    .await??;

    let analyzed = result.analyzed.into_iter().map(|a| AiSkillAnalysisResult {
        skillId: a.skill_id,
        skillName: a.skill_name,
        source: a.source,
        score: a.score,
        description: a.description,
        howToUse: a.how_to_use,
        reason: a.reason,
    }).collect();

    Ok(DeepSearchResult {
        thinking: result.thinking,
        totalFound: result.total_found,
        analyzed,
        searchStrategy: result.search_strategy,
        channelsUsed: result.channels_used,
        verificationPassed: result.verification_passed,
    })
}

/// Continue conversation-based search. Uses conversation history to refine results.
#[tauri::command]
pub async fn continue_ai_search(
    feedback: String,
    store: State<'_, Arc<SkillStore>>,
) -> Result<DeepSearchResult, AppError> {
    let ai_api_url = store
        .get_setting("ai_api_url")
        .map_err(AppError::db)?
        .filter(|v| !v.is_empty())
        .unwrap_or_default();
    let ai_api_key = store
        .get_setting("ai_api_key")
        .map_err(AppError::db)?
        .filter(|k| !k.is_empty())
        .ok_or_else(|| AppError::network(anyhow::anyhow!("AI API key not configured")))?;
    let proxy_url = store.proxy_url();

    // Get conversation history (last search query)
    let last_query = store
        .get_setting("last_search_query")
        .map_err(AppError::db)?
        .unwrap_or_else(|| "skills".to_string());

    let feedback_ref = feedback.clone();
    let proxy = proxy_url;
    let result = tauri::async_runtime::spawn_blocking(move || {
        ai_search_api::conversation_search(&ai_api_url, &ai_api_key, &last_query, &feedback_ref, proxy.as_deref())
            .map_err(AppError::network)
    })
    .await??;

    // Save feedback to conversation history
    let _ = store.set_setting("last_search_feedback", &feedback);

    let analyzed = result.analyzed.into_iter().map(|a| AiSkillAnalysisResult {
        skillId: a.skill_id,
        skillName: a.skill_name,
        source: a.source,
        score: a.score,
        description: a.description,
        howToUse: a.how_to_use,
        reason: a.reason,
    }).collect();

    Ok(DeepSearchResult {
        thinking: result.thinking,
        totalFound: result.total_found,
        analyzed,
        searchStrategy: result.search_strategy,
        channelsUsed: result.channels_used,
        verificationPassed: result.verification_passed,
    })
}
