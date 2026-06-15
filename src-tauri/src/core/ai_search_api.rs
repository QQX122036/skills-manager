use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use super::skillssh_api::{build_http_client, SkillsShSkill};

const DEFAULT_AI_API_URL: &str = "https://api.minimax.chat/v1";

#[derive(Debug, Clone, Serialize)]
pub struct SkillContent {
    pub skill_id: String,
    pub name: String,
    pub source: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSkillAnalysis {
    pub skill_id: String,
    pub skill_name: String,
    pub source: String,
    pub score: f64,
    pub description: String,
    pub how_to_use: String,
    pub reason: String,
}

#[derive(Debug, Serialize)]
pub struct DeepSearchResult {
    pub thinking: String,
    pub total_found: usize,
    pub analyzed: Vec<AiSkillAnalysis>,
    pub search_strategy: String,
    pub channels_used: Vec<String>,
    pub verification_passed: usize,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    temperature: f32,
    max_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    #[allow(dead_code)]
    choices: Vec<Choice>,
}

#[derive(Debug, Deserialize)]
struct Choice {
    #[allow(dead_code)]
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[allow(dead_code)]
    content: String,
}

/// Extract thinking content and keywords from the AI response text.
/// Returns (thinking_process, keywords).
/// Supports multiple formats: <think></think> tags, numbered lists, and plain keyword lines.
fn extract_thinking_and_keywords(text: &str) -> (String, Vec<String>) {
    log::info!("=== AI Response Analysis ===");
    log::info!("Raw response length: {} chars", text.len());
    log::debug!("Raw response:\n{}", text);
    
    let lines: Vec<&str> = text.lines().collect();
    let mut thinking_parts = Vec::new();
    let mut all_keywords: Vec<(usize, String)> = Vec::new();
    let mut in_thinking = false;
    let mut after_closing_tag = false;
    let mut has_thinking_tag = false;
    let mut has_closing_tag = false;

    for (idx, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        
        // Detect thinking tags
        if trimmed.contains("<think>") {
            in_thinking = true;
            has_thinking_tag = true;
            log::debug!("Found <think>at line {}", idx);
            continue;
        }
        if trimmed.contains("</think>") {
            in_thinking = false;
            after_closing_tag = true;
            has_closing_tag = true;
            log::debug!("Found </think> at line {}", idx);
            continue;
        }

        if in_thinking {
            thinking_parts.push(trimmed);
            continue;
        }

        // After </think>, collect keywords
        if after_closing_tag && !trimmed.is_empty() {
            let lower = trimmed.to_lowercase();
            
            // Skip numbered list markers like "1.", "2.", etc.
            let is_numbered_prefix = lower.starts_with("1.") || lower.starts_with("2.") || lower.starts_with("3.")
                || lower.starts_with("4.") || lower.starts_with("5.") || lower.starts_with("6.")
                || lower.starts_with("7.") || lower.starts_with("8.") || lower.starts_with("9.");
            
            let is_dash_prefix = trimmed.starts_with("-") || trimmed.starts_with("·");

            if is_numbered_prefix || is_dash_prefix {
                // Extract keyword after the marker
                let keyword = if is_numbered_prefix {
                    trimmed.trim_start_matches(|c: char| c.is_digit(10) || c == '.' || c == '-' || c == '·').trim()
                } else {
                    trimmed.trim_start_matches(|c: char| c == '-' || c == '·').trim()
                };
                
                // Clean up: remove explanations after colons or parentheses
                let clean = keyword
                    .split(':')
                    .next()
                    .unwrap_or(keyword)
                    .split('(')
                    .next()
                    .unwrap_or(keyword)
                    .trim();
                    
                if !clean.is_empty() && clean.len() > 1 && clean.len() < 60 {
                    log::debug!("Extracted keyword with marker: '{}'", clean);
                    all_keywords.push((idx, clean.to_string()));
                }
            } else {
                // Plain keyword line (no marker)
                // Skip lines that look like sentences (too long or contain sentence patterns)
                if trimmed.len() < 60 && !trimmed.contains('。') && !trimmed.contains('，') && !trimmed.contains('：') {
                    log::debug!("Extracted plain keyword: '{}'", trimmed);
                    all_keywords.push((idx, trimmed.to_string()));
                }
            }
        }
    }

    log::info!("Has thinking tag: {}, Has closing tag: {}, Keywords found: {}", has_thinking_tag, has_closing_tag, all_keywords.len());

    // If we have keywords from after the closing tag, use them
    if !all_keywords.is_empty() {
        // Deduplicate and limit to last 5 (most likely the actual keywords)
        let mut seen = std::collections::HashSet::new();
        let mut deduped = Vec::new();
        for (_, kw) in all_keywords.iter().rev() {
            let lower_kw = kw.to_lowercase();
            if !seen.contains(&lower_kw) && deduped.len() < 5 {
                seen.insert(lower_kw);
                deduped.push(kw.clone());
            }
        }
        deduped.reverse();
        log::info!("Extracted {} keywords from after closing tag: {:?}", deduped.len(), deduped);
        return (thinking_parts.join("\n"), deduped);
    }

    // Fallback: try to extract keywords from the thinking content itself
    // Look for patterns like "关键词:" or English words/phrases
    let thinking_text = thinking_parts.join("\n");
    let fallback_keywords = extract_fallback_keywords_from_thinking(&thinking_text);
    
    if !fallback_keywords.is_empty() {
        log::info!("Extracted {} keywords from thinking content: {:?}", fallback_keywords.len(), fallback_keywords);
        return (thinking_text, fallback_keywords);
    }

    // Last resort: extract English words from the entire response
    let mut english_keywords = Vec::new();
    let re = regex::Regex::new(r"[A-Za-z][A-Za-z\s\-]{2,30}").unwrap();
    for cap in re.find_iter(&text) {
        let word = cap.as_str().trim();
        let lower = word.to_lowercase();
        // Skip common English stop words and thinking-related words
        let skip_words = ["the", "and", "for", "you", "are", "this", "that", "with", "from", "have", "been", "will", "would", "should", "could", "need", "must", "can", "about", "after", "before", "between", "through", "during", "without", "within", "along", "following", "across", "behind", "beyond", "helps", "ensures", "provides", "includes", "thinking", "analysis", "process", "step", "steps", "example", "examples"];
        if !skip_words.contains(&lower.as_str()) && word.len() > 2 && word.len() < 40 {
            english_keywords.push(word.to_string());
        }
    }
    
    // Deduplicate
    let mut seen = std::collections::HashSet::new();
    let mut unique = Vec::new();
    for kw in &english_keywords {
        let lower = kw.to_lowercase();
        if !seen.contains(&lower) && unique.len() < 5 {
            seen.insert(lower);
            unique.push(kw.clone());
        }
    }

    log::info!("Extracted {} keywords from English words fallback: {:?}", unique.len(), unique);
    (thinking_text, unique)
}

/// Extract keywords from thinking content as a fallback when no explicit keywords are found.
/// This handles cases where MiniMax outputs thinking but no separate keyword section.
fn extract_fallback_keywords_from_thinking(thinking: &str) -> Vec<String> {
    let mut keywords = Vec::new();
    let mut in_numbered_list = false;

    for line in thinking.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            // blank line might separate sections
            if in_numbered_list && keywords.len() >= 3 {
                // We already have enough keywords, stop
                break;
            }
            continue;
        }
        let lower = trimmed.to_lowercase();

        // Detect numbered list items like "1. frontend design", "- frontend design"
        let is_numbered = lower.starts_with("1.") || lower.starts_with("2.") || lower.starts_with("3.")
            || lower.starts_with("4.") || lower.starts_with("5.") || lower.starts_with("6.")
            || lower.starts_with("7.") || lower.starts_with("8.") || lower.starts_with("9.")
            || lower.starts_with("-") || lower.starts_with("·");

        if is_numbered {
            in_numbered_list = true;
            let keyword = trimmed.trim_start_matches(|c: char| c.is_digit(10) || c == '.' || c == '-' || c == '·').trim();
            // Clean up keyword: take the first meaningful phrase before any explanation
            let clean_keyword = keyword
                .split(':')
                .next()
                .unwrap_or(keyword)
                .trim();
            // Remove parenthetical explanations like "(quantization)"
            let clean_keyword = clean_keyword
                .split('(')
                .next()
                .unwrap_or(clean_keyword)
                .trim();
            if !clean_keyword.is_empty() && clean_keyword.len() > 1 && clean_keyword.len() < 60 {
                keywords.push(clean_keyword.to_string());
            }
        }
    }

    // If we collected too many (because every numbered line was extracted),
    // take only the last 5 which are most likely the actual keywords
    if keywords.len() > 5 {
        keywords = keywords[keywords.len() - 5..].to_vec();
    }

    keywords
}

/// Call OpenAI-compatible API to expand user query into search keywords.
pub fn ai_expand_query_with_thinking(
    api_url: &str,
    api_key: &str,
    query: &str,
    proxy_url: Option<&str>,
) -> Result<(String, Vec<String>)> {
    let client = build_http_client(proxy_url, 30);

    let system_prompt = "你是一个 AI Agent 技能搜索助手。

## 任务
将用户的自然语言描述转换为 3-5 个简洁的英文搜索关键词,并用中文输出你的思考过程。

## 已知技能库 (供参考)
vercel-labs/skills: ai-web-design, browser-use, data-analysis, frontend-design, skill-creator, git-commit, playwright-test, skill-organizer, test-improver
anthropics/skills: research, code-review, skill-finder, skill-creator
microsoft/skills: office-automation, data-visualization, excel-automation

## 输出格式
你必须在 <think></think> 标签内输出中文思考过程,然后每行输出一个英文关键词。

示例:
<think>
用户的需求是关于量化分析,需要将数据转化为可度量的形式。这涉及数据分析、统计计算和指标评估等能力。相关的英文关键词包括:
</think>
quantification
data analysis
metrics
statistical analysis
calculation";

    let user_prompt = format!(
        "我需要一个能够实现以下功能的 skill:{}\n\n\
        请按照上述格式回复。",
        query
    );

    let request = ChatRequest {
        model: "MiniMax-M3".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        temperature: 0.7,
        max_tokens: 300,
    };

    let base_url = if api_url.is_empty() {
        DEFAULT_AI_API_URL
    } else {
        api_url
    };

    let request_url = format!("{}/chat/completions", base_url);
    log::info!("AI API request URL: {}", request_url);
    log::info!("AI API model: MiniMax-M3");

    let response = client
        .post(&request_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .context("Failed to call AI API")?;

    let status = response.status();
    let body_text = response
        .text()
        .unwrap_or_else(|_| "<unreadable>".to_string());

    log::info!("AI API response status: {}", status);
    log::debug!("AI API response body: {}", body_text);

    if !status.is_success() {
        let error_detail = format!(
            "AI API error ({}): {}",
            status,
            truncate_error_body(&body_text)
        );
        return Err(anyhow::anyhow!(error_detail));
    }

    let resp: serde_json::Value = serde_json::from_str(&body_text)
        .context("Failed to parse AI API response")?;

    let content = extract_response_content(&resp);
    log::info!("AI API extracted content length: {}", content.len());

    let (thinking, keywords) = extract_thinking_and_keywords(&content);
    if keywords.is_empty() {
        log::warn!("AI API returned empty content: {}", content.chars().take(200).collect::<String>());
        anyhow::bail!("AI API returned no keywords");
    }

    Ok((thinking, keywords))
}

/// Truncate error body for logging
fn truncate_error_body(body: &str) -> String {
    if body.len() > 500 {
        format!("{}...[truncated]", &body[..500])
    } else {
        body.to_string()
    }
}

/// Extract content from API response, supporting both OpenAI-compatible and MiniMax formats.
fn extract_response_content(resp: &serde_json::Value) -> String {
    if let Some(arr) = resp.get("choices").and_then(|c| c.as_array()) {
        if let Some(first) = arr.first() {
            if let Some(text) = first.get("message").and_then(|m| m.get("content")).and_then(|c| c.as_str()) {
                return text.to_string();
            }
            if let Some(text) = first.get("delta").and_then(|d| d.get("content")).and_then(|c| c.as_str()) {
                return text.to_string();
            }
            if let Some(messages) = first.get("messages").and_then(|m| m.as_array()) {
                if let Some(first_msg) = messages.first() {
                    if let Some(text) = first_msg.get("content").and_then(|c| c.as_str()) {
                        return text.to_string();
                    }
                    if let Some(text) = first_msg.get("text").and_then(|c| c.as_str()) {
                        return text.to_string();
                    }
                }
            }
        }
    }
    String::new()
}

/// AI-powered search: expand query via AI API, then search on skills.sh
pub fn search(
    api_url: &str,
    api_key: &str,
    query: &str,
    proxy_url: Option<&str>,
) -> Result<(String, Vec<SkillsShSkill>)> {
    // Step 1: Get keywords and thinking from AI
    let (thinking, keywords) = ai_expand_query_with_thinking(api_url, api_key, query, proxy_url)?;

    log::info!("AI expanded query '{}' into keywords: {:?}", query, keywords);
    if !thinking.is_empty() {
        log::info!("AI thinking: {}", thinking);
    }

    // Step 2: Search skills.sh with expanded keywords
    let client = build_http_client(proxy_url, 15);

    // Combine keywords into a search query
    let combined_query = keywords.join(" ");

    let url = format!(
        "https://skills.sh/api/search?q={}&limit=30",
        urlencoding::encode(&combined_query)
    );

    let resp: serde_json::Value = client
        .get(&url)
        .send()
        .context("Failed to search skills.sh")?
        .json()
        .context("Failed to parse search response")?;

    // Reuse the parsing logic from skillssh_api
    let skills = if let Some(arr) = resp.as_array() {
        parse_skills_array(arr)
    } else if let Some(arr) = resp.get("skills").and_then(|v| v.as_array()) {
        parse_skills_array(arr)
    } else {
        Vec::new()
    };

    Ok((thinking, skills))
}

fn parse_skills_array(arr: &[serde_json::Value]) -> Vec<SkillsShSkill> {
    let mut seen = std::collections::HashSet::new();
    let mut skills = Vec::new();

    for item in arr {
        let source = item
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let skill_id = item
            .get("skillId")
            .or_else(|| item.get("skill_id"))
            .or_else(|| item.get("id"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if source.is_empty() || skill_id.is_empty() {
            continue;
        }

        let id = format!("{}/{}", source, skill_id);
        if !seen.insert(id.clone()) {
            continue;
        }

        let name = item
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|v| !v.is_empty())
            .unwrap_or(&skill_id)
            .to_string();
        let installs = item.get("installs").and_then(|v| v.as_u64()).unwrap_or(0);

        skills.push(SkillsShSkill {
            id,
            skill_id,
            name,
            source,
            installs,
        });
    }

    skills
}

/// Fetch SKILL.md content from GitHub directly into memory (no disk write).
/// Tries main branch first, then master branch as fallback.
pub fn fetch_skill_content(
    source: &str,
    skill_id: &str,
    proxy_url: Option<&str>,
) -> Result<String> {
    let client = build_http_client(proxy_url, 10);

    // Try main branch first, then master branch
    let branches = ["main", "master"];
    let mut last_error = None;

    for branch in &branches {
        let raw_url = build_raw_github_url_with_branch(source, skill_id, branch)?;
        log::info!("Fetching SKILL.md from: {}", raw_url);

        match client
            .get(&raw_url)
            .timeout(std::time::Duration::from_secs(10))
            .send()
        {
            Ok(response) => {
                if response.status().is_success() {
                    let content = response.text().context("Failed to read SKILL.md content")?;
                    log::info!("Successfully fetched SKILL.md from {} branch", branch);
                    return Ok(content);
                } else {
                    log::debug!("SKILL.md fetch failed from {} branch: {}", branch, response.status());
                    last_error = Some(format!("Failed to fetch SKILL.md: {}", response.status()));
                }
            }
            Err(e) => {
                log::debug!("SKILL.md fetch error from {} branch: {}", branch, e);
                last_error = Some(e.to_string());
            }
        }
    }

    Err(anyhow::anyhow!(last_error.unwrap_or_else(|| "Failed to fetch SKILL.md from all branches".to_string())))
}

/// Build raw GitHub URL from skills.sh source format with specific branch.
fn build_raw_github_url_with_branch(source: &str, skill_id: &str, branch: &str) -> Result<String> {
    let clean_source = source.trim_start_matches('@');

    let parts: Vec<&str> = clean_source.split('/').collect();
    if parts.len() < 2 {
        anyhow::bail!("Invalid source format: {}", source);
    }

    let owner = parts[0];
    let repo = parts[1..].join("/");

    let url = format!(
        "https://raw.githubusercontent.com/{}/{}/{}/{}/SKILL.md",
        owner, repo, branch, skill_id
    );

    Ok(url)
}

/// Build raw GitHub URL from skills.sh source format (defaults to main branch).
fn build_raw_github_url(source: &str, skill_id: &str) -> Result<String> {
    build_raw_github_url_with_branch(source, skill_id, "main")
}

/// Multi-strategy search: search skills.sh with multiple queries and merge results.
fn multi_strategy_search(
    client: &reqwest::blocking::Client,
    user_query: &str,
    ai_keywords: &[String],
) -> Vec<SkillsShSkill> {
    let mut all_skills: Vec<SkillsShSkill> = Vec::new();
    let mut seen_ids = HashSet::new();

    // Strategy 1: Search with original user query (Chinese)
    log::info!("Strategy 1: Searching with original query: '{}'", user_query);
    let url = format!(
        "https://skills.sh/api/search?q={}&limit=20",
        urlencoding::encode(user_query)
    );
    match search_skills_sh_url(client, &url) {
        Ok(skills) => {
            log::info!("Strategy 1 returned {} skills", skills.len());
            for skill in skills {
                if seen_ids.insert(skill.skill_id.clone()) {
                    all_skills.push(skill);
                }
            }
        }
        Err(e) => log::warn!("Strategy 1 failed: {}", e),
    }

    // Strategy 2: Search with AI-generated keywords
    if !ai_keywords.is_empty() {
        let combined_query = ai_keywords.join(" ");
        log::info!("Strategy 2: Searching with AI keywords: '{}'", combined_query);
        let url = format!(
            "https://skills.sh/api/search?q={}&limit=20",
            urlencoding::encode(&combined_query)
        );
        match search_skills_sh_url(client, &url) {
            Ok(skills) => {
                log::info!("Strategy 2 returned {} skills", skills.len());
                for skill in skills {
                    if seen_ids.insert(skill.skill_id.clone()) {
                        all_skills.push(skill);
                    }
                }
            }
            Err(e) => log::warn!("Strategy 2 failed: {}", e),
        }

        // Strategy 3: Search each keyword individually
        log::info!("Strategy 3: Searching individual keywords");
        for kw in ai_keywords.iter().take(3) {
            let url = format!(
                "https://skills.sh/api/search?q={}&limit=15",
                urlencoding::encode(kw)
            );
            match search_skills_sh_url(client, &url) {
                Ok(skills) => {
                    log::info!("Strategy 3 (keyword '{}') returned {} skills", kw, skills.len());
                    for skill in skills {
                        if seen_ids.insert(skill.skill_id.clone()) {
                            all_skills.push(skill);
                        }
                    }
                }
                Err(e) => log::warn!("Strategy 3 (keyword '{}') failed: {}", kw, e),
            }
        }
    }

    log::info!("Multi-strategy search found {} unique skills total", all_skills.len());
    all_skills
}

/// Search skills.sh API from a constructed URL.
fn search_skills_sh_url(
    client: &reqwest::blocking::Client,
    url: &str,
) -> Result<Vec<SkillsShSkill>> {
    let resp: serde_json::Value = client
        .get(url)
        .timeout(std::time::Duration::from_secs(8))
        .send()
        .context("Failed to search skills.sh")?
        .json()
        .context("Failed to parse search response")?;

    let skills = if let Some(arr) = resp.as_array() {
        parse_skills_array(arr)
    } else if let Some(arr) = resp.get("skills").and_then(|v| v.as_array()) {
        parse_skills_array(arr)
    } else {
        Vec::new()
    };

    Ok(skills)
}

/// AI direct recommendation: ask AI to recommend skills based on its knowledge,
/// then verify they exist on skills.sh.
fn ai_direct_recommend_and_verify(
    query: &str,
    api_url: &str,
    api_key: &str,
    proxy_url: Option<&str>,
) -> Result<Vec<SkillsShSkill>> {
    let client = build_http_client(proxy_url, 30);

    let system_prompt = "你是一个 AI Agent 技能搜索专家，熟悉 skills.sh 平台上的各种技能。
当关键词搜索无法找到合适技能时，请根据用户需求直接推荐可能存在的技能。

请推荐 3-8 个 skills.sh 上可能存在的技能，格式为 JSON 数组:
[
  {\"skill_id\": \"skill-id\", \"source\": \"owner/repo\"},
  {\"skill_id\": \"skill-id-2\", \"source\": \"owner/repo\"}
]

已知常见技能包括:
- vercel-labs/skills: ai-web-design, browser-use, data-analysis, frontend-design, skill-creator, git-commit, playwright-test, skill-organizer, test-improver
- anthropics/skills: research, code-review, skill-finder, skill-creator
- microsoft/skills: office-automation, data-visualization, excel-automation

只返回 JSON 数组，不要其他内容。";

    let user_prompt = format!(
        "用户需求: {}\n\n请推荐 3-8 个可能符合该需求的 skills.sh 技能。",
        query
    );

    let base_url = if api_url.is_empty() {
        DEFAULT_AI_API_URL
    } else {
        api_url
    };

    let request_url = format!("{}/chat/completions", base_url);
    log::info!("AI direct recommendation URL: {}", request_url);

    let request = ChatRequest {
        model: "MiniMax-M3".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        temperature: 0.7,
        max_tokens: 2048,
    };

    let response = client
        .post(&request_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .context("Failed to call AI API for direct recommendation")?;

    let status = response.status();
    let body_text = response
        .text()
        .unwrap_or_else(|_| "<unreadable>".to_string());

    log::info!("AI direct recommendation response status: {}", status);

    if !status.is_success() {
        return Err(anyhow::anyhow!(
            "AI API error ({}): {}",
            status,
            truncate_error_body(&body_text)
        ));
    }

    let resp: serde_json::Value = serde_json::from_str(&body_text)
        .context("Failed to parse AI API response")?;

    let content = extract_response_content(&resp);

    // Parse AI recommendation: extract skill_id and source from JSON
    let json_str = extract_json_from_content(&content);
    log::info!("AI direct recommendation JSON: {}", json_str.chars().take(200).collect::<String>());

    #[derive(Debug, Deserialize)]
    struct SkillHint {
        skill_id: String,
        source: String,
    }

    let hints: Vec<SkillHint> = match serde_json::from_str(json_str) {
        Ok(h) => h,
        Err(e) => {
            log::warn!("Failed to parse AI direct recommendation: {}", e);
            return Ok(Vec::new());
        }
    };

    let total_hints = hints.len();

    // Verify each recommended skill exists on skills.sh
    let mut verified_skills = Vec::new();
    for hint in &hints {
        if verify_skill_exists(&hint.source, &hint.skill_id, proxy_url) {
            verified_skills.push(SkillsShSkill {
                id: format!("{}/{}", hint.source, hint.skill_id),
                skill_id: hint.skill_id.clone(),
                name: hint.skill_id.clone(),
                source: hint.source.clone(),
                installs: 0,
            });
        }
    }

    log::info!("AI direct recommendation: {} verified out of {} candidates", verified_skills.len(), total_hints);
    Ok(verified_skills)
}

/// Verify if a skill exists by trying to download its SKILL.md.
fn verify_skill_exists(
    source: &str,
    skill_id: &str,
    proxy_url: Option<&str>,
) -> bool {
    match fetch_skill_content(source, skill_id, proxy_url) {
        Ok(_) => true,
        Err(e) => {
            log::debug!("Skill verification failed for {}/{}: {}", source, skill_id, e);
            false
        }
    }
}


/// Fetch README.md from GitHub and extract a brief summary.
/// Tries main branch first, then master branch. Returns None if both fail.
fn fetch_readme_description(
    source: &str,
    proxy_url: Option<&str>,
) -> Option<String> {
    let client = build_http_client(proxy_url, 5);
    let clean_source = source.trim_start_matches('@');
    let parts: Vec<&str> = clean_source.split('/').collect();
    if parts.len() < 2 { return None; }

    let owner = parts[0];
    let repo = parts[1..].join("/");

    for branch in &["main", "master"] {
        let url = format!(
            "https://raw.githubusercontent.com/{}/{}/{}/README.md",
            owner, repo, branch
        );
        if let Ok(resp) = client.get(&url).timeout(std::time::Duration::from_secs(5)).send() {
            if resp.status().is_success() {
                if let Ok(raw) = resp.text() {
                    let summary = extract_readme_summary(&raw);
                    if !summary.is_empty() {
                        log::info!("README fetched for {}: {} chars", source, summary.len());
                        return Some(summary);
                    }
                }
            }
        }
    }
    None
}

/// Extract a meaningful description from README content.
/// Strategy: skip badges/code blocks, take the first meaningful paragraph after the first heading.
fn extract_readme_summary(content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut summary_lines: Vec<&str> = Vec::new();
    let mut in_code_block = false;
    let mut past_first_heading = false;

    for line in lines.iter().take(80) {
        let trimmed = line.trim();

        if trimmed.starts_with("```") { in_code_block = !in_code_block; continue; }
        if in_code_block { continue; }
        if trimmed.starts_with("[!") || trimmed.contains("shields.io") { continue; }
        if trimmed.starts_with("![") { continue; }

        if trimmed.starts_with('#') {
            past_first_heading = true;
            continue;
        }

        if trimmed.is_empty() {
            if !summary_lines.is_empty() { break; }
            continue;
        }

        if past_first_heading || !summary_lines.is_empty() {
            let cleaned = trimmed.replace('*', "").replace('_', "").replace('#', "").trim().to_string();
            if !cleaned.is_empty() {
                summary_lines.push(cleaned.leak());
            }
            if summary_lines.len() >= 3 { break; }
        }
    }

    if summary_lines.is_empty() { return String::new(); }

    let desc = summary_lines.join(" ");
    if desc.len() > 150 { format!("{}...", &desc[..150]) } else { desc }
}

/// Fetch SKILL.md from the skill subdirectory and extract description.
/// Tries main branch first, then master branch. Returns None if both fail.
fn fetch_skill_md_fallback_description(
    source: &str,
    skill_id: &str,
    _proxy_url: Option<&str>,
) -> Option<String> {
    let clean_source = source.trim_start_matches('@');
    let parts: Vec<&str> = clean_source.split('/').collect();
    if parts.len() < 2 { return None; }

    let owner = parts[0];
    let repo = parts[1..].join("/");

    for branch in &["main", "master"] {
        let url = format!(
            "https://raw.githubusercontent.com/{}/{}/{}/{}/SKILL.md",
            owner, repo, branch, skill_id
        );
        if let Ok(resp) = reqwest::blocking::Client::new()
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
        {
            if resp.status().is_success() {
                if let Ok(raw) = resp.text() {
                    let desc = extract_skill_md_description(&raw);
                    if !desc.is_empty() {
                        log::info!("SKILL.md fallback fetched for {}/{}: {} chars", source, skill_id, desc.len());
                        return Some(desc);
                    }
                }
            }
        }
    }
    None
}

/// Extract description from SKILL.md content.
/// Parses YAML frontmatter first, then markdown sections.
fn extract_skill_md_description(content: &str) -> String {
    // Try YAML frontmatter first
    if content.starts_with("---") {
        if let Some(end) = content[3..].find("---") {
            let yaml_block = &content[3..end + 3];
            // Extract name and description from YAML
            let mut name = String::new();
            let mut desc = String::new();
            for line in yaml_block.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("name:") {
                    name = trimmed["name:".len()..].trim().trim_matches('"').trim_matches('\'').to_string();
                }
                if trimmed.starts_with("description:") {
                    desc = trimmed["description:".len()..].trim().trim_matches('"').trim_matches('\'').to_string();
                }
            }
            if !desc.is_empty() {
                let result = if name.is_empty() { desc.clone() } else { format!("{} - {}", name, desc) };
                if result.len() > 150 { return format!("{}...", &result[..150]); }
                return result;
            }
        }
    }

    // Fallback: extract from markdown sections
    let lines: Vec<&str> = content.lines().collect();
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    let mut current_section = String::new();
    let mut current_content: Vec<String> = Vec::new();

    for line in lines.iter().take(50) {
        let trimmed = line.trim();
        if trimmed.starts_with("## ") || trimmed.starts_with("### ") {
            if !current_section.is_empty() {
                sections.push((current_section.clone(), current_content.clone()));
            }
            current_section = trimmed.trim_start_matches('#').trim().to_string();
            current_content.clear();
        } else if !trimmed.is_empty() && !trimmed.starts_with("---") && !trimmed.starts_with('`') {
            current_content.push(trimmed.replace('*', "").replace('_', "").trim().to_string());
            if current_content.len() >= 2 { break; }
        }
    }
    if !current_section.is_empty() {
        sections.push((current_section, current_content));
    }

    if sections.is_empty() {
        // Take first few non-empty lines
        let first_lines: Vec<&str> = lines.iter()
            .take(5)
            .filter(|l| !l.trim().is_empty() && !l.trim().starts_with("---"))
            .map(|l| l.trim())
            .collect();
        let result = first_lines.join(" ");
        if result.len() > 150 { return format!("{}...", &result[..150]); }
        return result;
    }

    let parts: Vec<String> = sections.iter()
        .take(2)
        .map(|(title, lines)| format!("{}: {}", title, lines.join(" ")))
        .collect();

    let result = parts.join("; ");
    if result.len() > 150 { format!("{}...", &result[..150]) } else { result }
}

/// Three-layer fallback description chain:
/// Layer 1: README.md (repository level overview)
/// Layer 2: SKILL.md (skill-level detail)
/// Layer 3: Name-based inference (last resort)
fn fetch_skill_description(
    source: &str,
    skill_id: &str,
    name: &str,
    proxy_url: Option<&str>,
) -> String {
    // Layer 1: Try README.md
    if let Some(desc) = fetch_readme_description(source, proxy_url) {
        log::info!("Fallback L1 (README) for {}/{}: OK", source, skill_id);
        return desc;
    }

    // Layer 2: Try SKILL.md (individual retry, may succeed where batch failed)
    if let Some(desc) = fetch_skill_md_fallback_description(source, skill_id, proxy_url) {
        log::info!("Fallback L2 (SKILL.md) for {}/{}: OK", source, skill_id);
        return desc;
    }

    // Layer 3: Name inference (last resort)
    log::info!("Fallback L3 (name inference) for {}/{}", source, skill_id);
    infer_description_from_name(skill_id, name)
}

/// Infer a description for a skill based on its name and skill ID.
fn infer_description_from_name(skill_id: &str, name: &str) -> String {
    let combined = format!("{} {}", skill_id, name).to_lowercase();

    let tech_descs = [
        ("playwright", "基于 Playwright 的浏览器自动化测试技能，支持 UI 测试、E2E 测试和页面交互"),
        ("puppeteer", "基于 Puppeteer 的无头浏览器控制技能，支持网页截图、PDF 生成和自动化操作"),
        ("selenium", "基于 Selenium 的跨浏览器自动化测试框架，支持多语言多浏览器"),
        ("cypress", "基于 Cypress 的前端端到端测试技能，支持实时调试和时间旅行"),
        ("browser", "浏览器相关技能，涉及浏览器自动化、网页抓取或浏览器插件开发"),
        ("automation", "自动化相关技能，可自动执行重复性任务或工作流程"),
        ("testing", "测试相关技能，帮助编写、运行和管理自动化测试"),
        ("test", "测试工具技能，提供测试框架、测试用例或测试报告功能"),
        ("e2e", "端到端测试技能，模拟真实用户操作进行完整流程测试"),
        ("visual", "视觉测试相关，涉及截图对比、UI 回归测试或视觉验证"),
        ("qa", "质量保证相关，帮助进行软件质量检查和自动化验证"),
        ("web", "Web 开发相关技能，涉及前端构建、Web 服务或网站开发"),
        ("react", "React 生态技能，涉及 React 组件、Hooks 或状态管理"),
        ("vue", "Vue.js 生态技能，涉及 Vue 组件、指令或状态管理"),
        ("next", "Next.js 全栈框架技能，支持 SSR、SSG 和 API 路由"),
        ("tailwind", "Tailwind CSS 实用类框架技能，支持快速样式开发"),
        ("api", "API 相关技能，涉及 API 设计、测试或文档生成"),
        ("cli", "命令行工具技能，提供终端交互和批处理功能"),
        ("git", "Git 版本控制相关技能，涉及分支管理、代码审查或工作流"),
        ("docker", "Docker 容器化技能，支持容器编排、镜像构建或部署"),
        ("kubernetes", "Kubernetes 编排技能，涉及集群管理、服务部署或监控"),
        ("aws", "AWS 云服务技能，支持 S3、Lambda、EC2 等云资源管理"),
        ("database", "数据库相关技能，涉及 SQL 查询、迁移或优化"),
        ("graphql", "GraphQL API 技能，支持查询构建、Schema 设计或代码生成"),
        ("typescript", "TypeScript 类型系统技能，涉及类型定义、代码检查或编译"),
        ("python", "Python 编程技能，涉及脚本编写、数据分析或 Web 开发"),
        ("rust", "Rust 系统编程技能，涉及内存安全、并发或性能优化"),
        ("go", "Go 语言技能，支持并发编程、微服务或 CLI 工具开发"),
        ("node", "Node.js 运行时技能，涉及后端服务、API 或中间件开发"),
        ("agent", "AI Agent 技能，支持智能代理、自主决策或工具调用"),
        ("claude", "Claude AI 相关技能，涉及提示工程、工具集成或对话管理"),
        ("cursor", "Cursor 编辑器技能，涉及代码生成、智能补全或项目导航"),
        ("copilot", "GitHub Copilot 相关技能，支持 AI 辅助编程和代码建议"),
        ("skill", "技能管理相关，涉及技能的创建、安装、同步或组织"),
    ];

    let mut found_descs: Vec<&str> = Vec::new();
    for (keyword, desc) in tech_descs.iter() {
        if combined.contains(keyword) && !found_descs.contains(desc) {
            found_descs.push(desc);
        }
    }

    if !found_descs.is_empty() {
        let desc = found_descs[..std::cmp::min(2, found_descs.len())].join("；");
        return desc;
    }

    if name.contains('-') {
        let parts: Vec<&str> = name.split('-').collect();
        let tech_part = parts.iter().find(|p| {
            matches!(**p, "playwright" | "puppeteer" | "selenium" | "cypress" | "react" | "vue" | "next" | "tailwind" | "docker" | "rust" | "python")
        });
        if let Some(tech) = tech_part {
            return format!("与 {} 相关的技能，可从名称推断涉及该技术的特定功能模块", tech);
        }
    }

    format!("技能库中的候选技能，涉及 {} 相关功能", name)
}


/// Calculate an enhanced score based on installs, keyword matching, AND README/content relevance.
fn calculate_enhanced_score(
    skill_id: &str,
    name: &str,
    installs: usize,
    keywords: &[String],
    content_summary: Option<&str>,
    user_query: &str,
) -> f64 {
    // 1. Install-based score (max 4.0)
    let install_score = if installs > 10000 { 4.0 }
    else if installs > 5000 { 3.5 }
    else if installs > 1000 { 3.0 }
    else if installs > 500 { 2.5 }
    else if installs > 100 { 2.0 }
    else { 1.5 };

    // 2. Keyword match bonus (max 2.0)
    let combined = format!("{} {}", skill_id, name).to_lowercase();
    let keyword_bonus: f64 = keywords.iter()
        .filter(|kw| combined.contains(&kw.to_lowercase()))
        .count() as f64 * 0.5;

    // 3. Content relevance bonus (max 3.0) - from README or SKILL.md summary
    let content_bonus = if let Some(summary) = content_summary {
        let summary_lower = summary.to_lowercase();
        let query_lower = user_query.to_lowercase();

        // Check overlap between query words and content
        let query_words: Vec<&str> = query_lower.split_whitespace()
            .filter(|w| w.len() > 1)
            .collect();

        let match_count = query_words.iter()
            .filter(|w| summary_lower.contains(*w))
            .count();

        (match_count as f64 * 0.6).min(3.0)
    } else {
        0.0
    };

    (install_score + keyword_bonus + content_bonus).min(10.0)
}

/// Generate enhanced recommendation reason based on content and installs.
fn infer_recommendation_reason_enhanced(
    query: &str,
    skill_id: &str,
    name: &str,
    installs: usize,
    keywords: &[String],
    content_summary: Option<&str>,
) -> String {
    let install_desc = if installs > 10000 { "社区认可度极高" }
    else if installs > 5000 { "社区认可度高" }
    else if installs > 1000 { "热门" }
    else if installs > 500 { "较受欢迎" }
    else if installs > 100 { "有一定用户基础" }
    else { "小众但有潜力" };

    let combined = format!("{} {}", skill_id, name).to_lowercase();

    // Check for keyword matches
    let matched_keywords: Vec<&str> = keywords.iter()
        .filter(|kw| combined.contains(&kw.to_lowercase()))
        .map(|s| s.as_str())
        .collect();

    // If we have content summary (from README or SKILL.md), use it
    if let Some(summary) = content_summary {
        let summary_lower = summary.to_lowercase();
        let query_lower = query.to_lowercase();

        // Find concepts from the summary that match the query
        let query_words: Vec<&str> = query_lower.split_whitespace()
            .filter(|w| w.len() > 1)
            .collect();

        let matched_concepts: Vec<&str> = query_words.iter()
            .filter(|w| summary_lower.contains(*w))
            .map(|s| *s)
            .collect();

        if !matched_concepts.is_empty() {
            let concepts_str = matched_concepts.join("、");
            let brief = if summary.len() > 60 { &summary[..60] } else { summary };
            return format!(
                "简介：{}；与搜索需求匹配（「{}」）；{}技能（{} 次安装）",
                brief, concepts_str, install_desc, installs
            );
        }

        // Content exists but low direct relevance - show it anyway
        let brief = if summary.len() > 80 { &summary[..80] } else { summary };
        return format!(
            "简介：{}；{}（{} 次安装），建议查看详情确认",
            brief, install_desc, installs
        );
    }

    // Fallback to keyword-based reason
    if !matched_keywords.is_empty() {
        let kw_str = matched_keywords.join("、");
        format!("与搜索意图匹配，涵盖「{}」关键词；{}技能（{} 次安装）", kw_str, install_desc, installs)
    } else {
        let query_lower = query.to_lowercase();
        let query_words: Vec<&str> = query_lower.split_whitespace().collect();
        let matched_query: Vec<&&str> = query_words.iter()
            .filter(|w| w.len() > 1 && combined.contains(*w))
            .collect();

        if !matched_query.is_empty() {
            let w_str = matched_query.iter().map(|s| **s).collect::<Vec<_>>().join("、");
            format!("名称含「{}」关键词；{}（{} 次安装）", w_str, install_desc, installs)
        } else {
            format!("{}技能（{} 次安装）；建议查看详情确认是否符合需求", install_desc, installs)
        }
    }
}


/// Generate a recommendation reason based on the user query and skill metadata.
fn infer_recommendation_reason(query: &str, skill_id: &str, name: &str, installs: usize, keywords: &[String]) -> String {
    let combined = format!("{} {}", skill_id, name).to_lowercase();

    let matched_keywords: Vec<&str> = keywords.iter()
        .filter(|kw| combined.contains(&kw.to_lowercase()))
        .map(|s| s.as_str())
        .collect();

    let install_desc = if installs > 1000 {
        format!("热门技能（{} 次安装）", installs)
    } else if installs > 500 {
        format!("较受欢迎（{} 次安装）", installs)
    } else if installs > 100 {
        format!("有一定用户基础（{} 次安装）", installs)
    } else {
        format!("{} 次安装", installs)
    };

    if !matched_keywords.is_empty() {
        let kw_str = matched_keywords.join("、");
        return format!("与搜索意图高度匹配，涵盖「{}」等关键词；{}", kw_str, install_desc);
    }

    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower.split_whitespace().collect();
    let matched_query_words: Vec<&&str> = query_words.iter()
        .filter(|w| w.len() > 1 && combined.contains(*w))
        .collect();

    if !matched_query_words.is_empty() {
        let w_str = matched_query_words.iter().map(|s| **s).collect::<Vec<_>>().join("、");
        return format!("名称中包含「{}」关键词，与搜索需求相关；{}", w_str, install_desc);
    }

    format!("候选技能，{}；建议查看详情以确认是否符合需求", install_desc)
}

/// Calculate a score for unanalyzed skills based on installs and keyword matching.
fn calculate_fallback_score(skill_id: &str, name: &str, installs: usize, keywords: &[String]) -> f64 {
    let combined = format!("{} {}", skill_id, name).to_lowercase();

    let install_score = if installs > 5000 {
        9.0
    } else if installs > 1000 {
        8.0
    } else if installs > 500 {
        7.0
    } else if installs > 100 {
        6.0
    } else {
        5.0
    };

    let keyword_bonus: f64 = keywords.iter()
        .filter(|kw| combined.contains(&kw.to_lowercase()))
        .count() as f64 * 0.5;

    (install_score + keyword_bonus).min(9.5)
}

/// Deep search: expand query, fetch candidates, download SKILL.md, analyze with AI.
/// Implements a 4-layer progressive search strategy:
/// - Layer 1: AI keyword expansion
/// - Layer 2: Multi-strategy skills.sh search
/// - Layer 3: AI direct recommendation + verification
/// - Layer 4: SKILL.md download + AI analysis
pub fn deep_search(
    api_url: &str,
    api_key: &str,
    query: &str,
    proxy_url: Option<&str>,
) -> Result<DeepSearchResult> {
    let client = build_http_client(proxy_url, 15);

    log::info!("=== Deep search starting for query: {} ===", query);

    // Layer 1: AI expands query into keywords
    let (thinking, keywords) = ai_expand_query_with_thinking(api_url, api_key, query, proxy_url)?;
    log::info!("Layer 1 - AI expanded query into {} keywords: {:?}", keywords.len(), keywords);

    // Layer 2: Multi-strategy skills.sh search
    log::info!("Layer 2 - Starting multi-strategy search");
    let skills = multi_strategy_search(&client, query, &keywords);
    let mut channels_used = vec!["skills.sh".to_string()];
    let mut search_strategy = "multi_strategy".to_string();

    // Layer 3: Fallback to AI direct recommendation if no skills found
    let mut skills = skills;
    if skills.is_empty() {
        log::info!("Layer 2 found no results, falling back to Layer 3: AI direct recommendation");
        channels_used.push("ai_direct".to_string());
        search_strategy = "ai_direct_recommendation".to_string();

        match ai_direct_recommend_and_verify(query, api_url, api_key, proxy_url) {
            Ok(recommended) => {
                skills = recommended;
                log::info!("Layer 3 - AI direct recommendation found {} verified skills", skills.len());
                
                // If AI direct recommendation also found nothing, try a broader search
                if skills.is_empty() {
                    log::info!("Layer 3 also found no results, trying broader keyword search");
                    channels_used.push("broad_search".to_string());
                    search_strategy = "broad_keyword_search".to_string();
                    
                    // Try searching with each keyword individually with higher limits
                    for kw in keywords.iter() {
                        let url = format!(
                            "https://skills.sh/api/search?q={}&limit=30",
                            urlencoding::encode(kw)
                        );
                        match search_skills_sh_url(&client, &url) {
                            Ok(broad_skills) => {
                                log::info!("Broad search for '{}' found {} skills", kw, broad_skills.len());
                                for skill in broad_skills {
                                    let skill_id_key = format!("{}/{}", skill.source, skill.skill_id);
                                    let mut seen_ids = HashSet::new();
                                    if seen_ids.insert(skill_id_key.clone()) {
                                        skills.push(skill);
                                    }
                                }
                            }
                            Err(e) => log::warn!("Broad search for '{}' failed: {}", kw, e),
                        }
                    }
                }
            }
            Err(e) => {
                log::warn!("Layer 3 - AI direct recommendation failed: {}", e);
                // Even if AI recommendation fails, try broad keyword search as last resort
                channels_used.push("broad_search".to_string());
                search_strategy = "broad_keyword_search".to_string();
                
                for kw in keywords.iter() {
                    let url = format!(
                        "https://skills.sh/api/search?q={}&limit=30",
                        urlencoding::encode(kw)
                    );
                    match search_skills_sh_url(&client, &url) {
                        Ok(broad_skills) => {
                            log::info!("Broad search for '{}' found {} skills", kw, broad_skills.len());
                            for skill in broad_skills {
                                let skill_id_key = format!("{}/{}", skill.source, skill.skill_id);
                                let mut seen_ids = HashSet::new();
                                if seen_ids.insert(skill_id_key.clone()) {
                                    skills.push(skill);
                                }
                            }
                        }
                        Err(e) => log::warn!("Broad search for '{}' failed: {}", kw, e),
                    }
                }
            }
        }
    }

    let total_found = skills.len();
    log::info!("Total candidate skills after Layer 2/3: {}", total_found);

    if skills.is_empty() {
        return Ok(DeepSearchResult {
            thinking,
            total_found: 0,
            analyzed: Vec::new(),
            search_strategy,
            channels_used,
            verification_passed: 0,
        });
    }

    // Layer 4: Download SKILL.md and AI analysis
    log::info!("Layer 4 - Fetching SKILL.md content and analyzing");
    let top_skills: Vec<_> = skills.into_iter().take(10).collect();

    let skill_contents: Vec<SkillContent> = top_skills
        .iter()
        .filter_map(|s| {
            match fetch_skill_content(&s.source, &s.skill_id, proxy_url) {
                Ok(content) => Some(SkillContent {
                    skill_id: s.skill_id.clone(),
                    name: s.name.clone(),
                    source: s.source.clone(),
                    content,
                }),
                Err(e) => {
                    log::warn!("Failed to fetch SKILL.md for {}/{}: {}", s.source, s.skill_id, e);
                    None
                }
            }
        })
        .collect();

    log::info!("Successfully fetched {} SKILL.md files out of {} candidates", skill_contents.len(), top_skills.len());

    // If we have some SKILL.md content, proceed with AI analysis
    if !skill_contents.is_empty() {
        let mut analyzed = analyze_skills(query, &skill_contents, api_url, api_key, proxy_url)?;

        // Fill in skill_name and source from the fetched contents
        for analysis in &mut analyzed {
            if analysis.skill_name.is_empty() || analysis.source.is_empty() {
                if let Some(sc) = skill_contents.iter().find(|s| s.skill_id == analysis.skill_id) {
                    if analysis.skill_name.is_empty() {
                        analysis.skill_name = sc.name.clone();
                    }
                    if analysis.source.is_empty() {
                        analysis.source = sc.source.clone();
                    }
                }
            }
        }

        log::info!("=== Deep search complete: {} recommendations ===", analyzed.len());

        Ok(DeepSearchResult {
            thinking,
            total_found,
            analyzed,
            search_strategy,
            channels_used,
            verification_passed: skill_contents.len(),
        })
    } else {
        // Fallback: return raw search results with enhanced descriptions
        log::info!("Layer 4 - SKILL.md fetch failed, using 3-layer fallback chain for descriptions");
        channels_used.push("raw_fallback".to_string());
        search_strategy = format!("{}_with_smart_fallback", search_strategy);

        let fallback_analyses: Vec<AiSkillAnalysis> = top_skills.iter().map(|s| {
            // Use 3-layer fallback chain: README -> SKILL.md -> name inference
            let description = fetch_skill_description(&s.source, &s.skill_id, &s.name, proxy_url);

            // Try to get content summary for enhanced scoring
            let content_summary = fetch_readme_description(&s.source, proxy_url)
                .or_else(|| fetch_skill_md_fallback_description(&s.source, &s.skill_id, proxy_url));

            let install_count = s.installs.try_into().unwrap_or(0usize);
            let score = calculate_enhanced_score(&s.skill_id, &s.name, install_count, &keywords, content_summary.as_deref(), query);
            let reason = infer_recommendation_reason_enhanced(query, &s.skill_id, &s.name, install_count, &keywords, content_summary.as_deref());

            AiSkillAnalysis {
                skill_id: s.skill_id.clone(),
                skill_name: s.name.clone(),
                source: s.source.clone(),
                score,
                description,
                how_to_use: "点击查看详情或安装后使用".to_string(),
                reason,
            }
        }).collect();

        log::info!("=== Deep search smart fallback complete: {} results ===", fallback_analyses.len());

        Ok(DeepSearchResult {
            thinking: format!("{}

[提示] 使用智能兜底链生成描述（README → SKILL.md → 名称推断）", thinking),
            total_found,
            analyzed: fallback_analyses,
            search_strategy,
            channels_used,
            verification_passed: 0,
        })
    }
}

/// Conversation-based search: refine results based on user feedback.
/// Supports: "换一批" (get more), "太专业了" (simpler), "要更简单的" (easier), etc.
pub fn conversation_search(
    api_url: &str,
    api_key: &str,
    original_query: &str,
    feedback: &str,
    previous_skill_ids: &[String],
    conversation_history: Option<&str>,
    proxy_url: Option<&str>,
) -> Result<DeepSearchResult> {
    log::info!("=== Conversation search: original='{}', feedback='{}', previous_count={}, has_history={} ===", original_query, feedback, previous_skill_ids.len(), conversation_history.is_some());

    let client = build_http_client(proxy_url, 15);

    // Analyze feedback to determine search strategy adjustment
    let feedback_lower = feedback.to_lowercase();
    let (strategy_adjustment, keywords_override) = if feedback_lower.contains("换一批") || feedback_lower.contains("更多") {
        // User wants more options - use different keywords or broader search
        ("user_wants_more_options".to_string(), vec![])
    } else if feedback_lower.contains("太专业") || feedback_lower.contains("太难") || feedback_lower.contains("简单") {
        // User wants simpler skills
        ("user_wants_simpler_skills".to_string(), vec!["basic".to_string(), "simple".to_string(), "easy".to_string(), "beginner".to_string()])
    } else if feedback_lower.contains("高级") || feedback_lower.contains("复杂") || feedback_lower.contains("专业") {
        // User wants more advanced skills
        ("user_wants_advanced_skills".to_string(), vec!["advanced".to_string(), "expert".to_string(), "professional".to_string()])
    } else {
        // Custom feedback - let AI handle it
        ("custom_feedback".to_string(), vec![])
    };

    let thinking;
    let skills;

    // If we have keyword overrides, search directly
    if !keywords_override.is_empty() {
        log::info!("Using keyword override for simpler/advanced search");
        thinking = format!("根据反馈'{}'，使用预设关键词搜索", feedback);
        skills = multi_strategy_search(&client, original_query, &keywords_override);
    } else {
        // Use AI to refine the search based on feedback, passing previous results to avoid duplicates
        log::info!("Using AI to refine search based on feedback");
        match refine_query_with_feedback(api_url, api_key, original_query, feedback, previous_skill_ids, conversation_history, proxy_url) {
            Ok((t, refined_keywords)) => {
                thinking = t;
                skills = multi_strategy_search(&client, original_query, &refined_keywords);
            }
            Err(e) => {
                log::warn!("Failed to refine query: {}", e);
                thinking = format!("反馈处理失败 ({}), 使用原始搜索", e);
                skills = multi_strategy_search(&client, original_query, &[]);
            }
        }
    }

    let mut channels_used = vec!["skills.sh".to_string()];
    let search_strategy = format!("conversation_{}", strategy_adjustment);

    let total_found = skills.len();
    log::info!("Conversation search found {} candidates", total_found);

    if skills.is_empty() {
        return Ok(DeepSearchResult {
            thinking,
            total_found: 0,
            analyzed: Vec::new(),
            search_strategy,
            channels_used,
            verification_passed: 0,
        });
    }

    // Download SKILL.md and analyze
    log::info!("Layer 4 - Fetching SKILL.md content for conversation search");
    let top_skills: Vec<_> = skills.into_iter().take(10).collect();

    let skill_contents: Vec<SkillContent> = top_skills
        .iter()
        .filter_map(|s| {
            match fetch_skill_content(&s.source, &s.skill_id, proxy_url) {
                Ok(content) => Some(SkillContent {
                    skill_id: s.skill_id.clone(),
                    name: s.name.clone(),
                    source: s.source.clone(),
                    content,
                }),
                Err(e) => {
                    log::warn!("Failed to fetch SKILL.md for {}: {}", s.skill_id, e);
                    None
                }
            }
        })
        .collect();

    log::info!("Conversation search: fetched {} SKILL.md out of {} candidates", skill_contents.len(), top_skills.len());

    if !skill_contents.is_empty() {
        // Use feedback-aware analysis prompt
        let mut analyzed = analyze_skills_with_feedback(original_query, feedback, &skill_contents, api_url, api_key, proxy_url)?;

        // Fill in skill_name and source
        for analysis in &mut analyzed {
            if analysis.skill_name.is_empty() || analysis.source.is_empty() {
                if let Some(sc) = skill_contents.iter().find(|s| s.skill_id == analysis.skill_id) {
                    if analysis.skill_name.is_empty() {
                        analysis.skill_name = sc.name.clone();
                    }
                    if analysis.source.is_empty() {
                        analysis.source = sc.source.clone();
                    }
                }
            }
        }

        log::info!("=== Conversation search complete: {} recommendations ===", analyzed.len());

        Ok(DeepSearchResult {
            thinking,
            total_found,
            analyzed,
            search_strategy,
            channels_used,
            verification_passed: skill_contents.len(),
        })
    } else {
        // Fallback: return raw search results with enhanced descriptions
        log::info!("Conversation search: SKILL.md fetch failed, using 3-layer fallback chain");
        channels_used.push("raw_fallback".to_string());

        let fallback_analyses: Vec<AiSkillAnalysis> = top_skills.iter().map(|s| {
            let description = fetch_skill_description(&s.source, &s.skill_id, &s.name, proxy_url);
            let content_summary = fetch_readme_description(&s.source, proxy_url)
                .or_else(|| fetch_skill_md_fallback_description(&s.source, &s.skill_id, proxy_url));

            let install_count = s.installs.try_into().unwrap_or(0usize);
            let score = calculate_enhanced_score(&s.skill_id, &s.name, install_count, &[], content_summary.as_deref(), original_query);
            let reason = infer_recommendation_reason_enhanced(original_query, &s.skill_id, &s.name, install_count, &[], content_summary.as_deref());

            AiSkillAnalysis {
                skill_id: s.skill_id.clone(),
                skill_name: s.name.clone(),
                source: s.source.clone(),
                score,
                description,
                how_to_use: "点击查看详情或安装后使用".to_string(),
                reason,
            }
        }).collect();

        log::info!("=== Conversation search smart fallback complete: {} results ===", fallback_analyses.len());

        Ok(DeepSearchResult {
            thinking: format!("{}

[提示] 使用智能兜底链生成描述（README → SKILL.md → 名称推断）", thinking),
            total_found,
            analyzed: fallback_analyses,
            search_strategy,
            channels_used,
            verification_passed: 0,
        })
    }
}

/// Refine search query based on user feedback using AI.
fn refine_query_with_feedback(
    api_url: &str,
    api_key: &str,
    original_query: &str,
    feedback: &str,
    previous_skill_ids: &[String],
    conversation_history: Option<&str>,
    proxy_url: Option<&str>,
) -> Result<(String, Vec<String>)> {
    let client = build_http_client(proxy_url, 15);

    let system_prompt = "你是一个 AI 技能搜索优化专家。用户之前搜索了某个需求，现在给出了反馈。
请根据反馈调整搜索策略，生成新的关键词。

反馈类型:
- '换一批'/'更多': 用户想要更多选项 → 生成不同但相关的关键词，必须避开已推荐过的技能
- '太专业'/'太难'/'简单': 用户想要更简单的技能 → 生成基础/入门关键词
- '高级'/'复杂'/'专业': 用户想要更专业的技能 → 生成高级关键词
- 其他: 根据具体反馈调整

重要：如果用户反馈是'换一批'，你必须避开已推荐过的技能，生成全新的、不同角度的关键词。

输出格式:
<think>
[中文思考过程]
</think>
keyword1
keyword2
keyword3

只返回思考过程和关键词，不要其他内容。";

    let previous_context = if previous_skill_ids.is_empty() {
        String::new()
    } else {
        let skill_list = previous_skill_ids.join(", ");
        format!("\n已推荐过的技能（请避免重复）: {}\n", skill_list)
    };

    let conversation_history_context = if let Some(history) = conversation_history {
        format!("\n\n对话历史:\n{}\n\n重要：请根据对话历史理解用户意图，如果反馈是'换一批'，请避开上述已推荐的技能。", history)
    } else {
        String::new()
    };

    let user_prompt = format!(
        "原始需求: {}{}{}\n用户反馈: {}\n\n请根据反馈和对话历史调整搜索关键词，生成 3-5 个全新的英文搜索关键词。",
        original_query, previous_context, conversation_history_context, feedback
    );

    let base_url = if api_url.is_empty() {
        DEFAULT_AI_API_URL
    } else {
        api_url
    };

    let request_url = format!("{}/chat/completions", base_url);

    let request = ChatRequest {
        model: "MiniMax-M3".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        temperature: 0.7,
        max_tokens: 512,
    };

    let response = client
        .post(&request_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .context("Failed to call AI API for query refinement")?;

    let status = response.status();
    let body_text = response.text().unwrap_or_else(|_| "<unreadable>".to_string());

    if !status.is_success() {
        return Err(anyhow::anyhow!(
            "AI API error ({}): {}",
            status,
            truncate_error_body(&body_text)
        ));
    }

    let resp: serde_json::Value = serde_json::from_str(&body_text)
        .context("Failed to parse AI API response")?;

    let content = extract_response_content(&resp);
    let (thinking, keywords) = extract_thinking_and_keywords(&content);

    Ok((thinking, keywords))
}

/// Analyze skills with feedback context for conversation search.
fn analyze_skills_with_feedback(
    user_query: &str,
    feedback: &str,
    skills: &[SkillContent],
    api_url: &str,
    api_key: &str,
    proxy_url: Option<&str>,
) -> Result<Vec<AiSkillAnalysis>> {
    let client = build_http_client(proxy_url, 60);

    let mut skills_section = String::new();
    for (i, skill) in skills.iter().enumerate() {
        skills_section.push_str(&format!(
            "技能 {}：{}\n来源：{}\n内容：\n{}\n\n",
            i + 1,
            skill.name,
            skill.source,
            skill.content
        ));
    }

    let feedback_text = feedback.to_string();
    let feedback_context = format!(
        "用户反馈: {}。请根据反馈调整评分和推荐理由。",
        feedback_text
    );

    let system_prompt = "你是一个 AI 技能评估专家。请分析以下技能,根据用户需求和反馈进行评分和描述。
重要:所有描述字段必须使用中文回答。

请按以下 JSON 格式返回分析结果(按相关性从高到低排序):
[
  {
    \"skill_id\": \"技能 ID\",
    \"skill_name\": \"技能名称\",
    \"source\": \"来源仓库\",
    \"score\": 9.2,
    \"description\": \"用中文描述这个技能是什么,能做什么\",
    \"how_to_use\": \"用中文简要说明使用方式\",
    \"reason\": \"用中文说明为什么推荐(考虑用户反馈)\"
  }
]

评分标准:
- 9-10 分:完美匹配用户需求
- 7-8 分:高度相关
- 5-6 分:部分相关
- 1-4 分:关联度较低

注意:只返回 JSON 数组,不要其他内容。description、how_to_use、reason 字段必须使用中文。";

    let user_prompt = format!(
        "用户需求:{}\n\n{}\n\n待分析技能:\n\n{}",
        user_query, feedback_context, skills_section
    );

    let base_url = if api_url.is_empty() {
        DEFAULT_AI_API_URL
    } else {
        api_url
    };

    let request_url = format!("{}/chat/completions", base_url);
    log::info!("AI conversation analysis URL: {}", request_url);

    let request = ChatRequest {
        model: "MiniMax-M3".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        temperature: 0.7,
        max_tokens: 4096,
    };

    let response = client
        .post(&request_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .context("Failed to call AI API for conversation analysis")?;

    let status = response.status();
    let body_text = response
        .text()
        .unwrap_or_else(|_| "<unreadable>".to_string());

    log::info!("AI conversation analysis response status: {}", status);

    if !status.is_success() {
        let error_detail = format!(
            "AI API error ({}): {}",
            status,
            truncate_error_body(&body_text)
        );
        return Err(anyhow::anyhow!(error_detail));
    }

    let resp: serde_json::Value = serde_json::from_str(&body_text)
        .context("Failed to parse AI API response")?;

    let content = extract_response_content(&resp);
    log::info!("AI conversation analysis extracted content length: {}", content.len());

    let analyses = parse_ai_analysis_json(&content)?;
    Ok(analyses)
}

/// Batch analyze skills with AI, returns structured results with Chinese output.
pub fn analyze_skills(
    user_query: &str,
    skills: &[SkillContent],
    api_url: &str,
    api_key: &str,
    proxy_url: Option<&str>,
) -> Result<Vec<AiSkillAnalysis>> {
    let client = build_http_client(proxy_url, 60);

    let mut skills_section = String::new();
    for (i, skill) in skills.iter().enumerate() {
        skills_section.push_str(&format!(
            "技能 {}：{}\n来源：{}\n内容：\n{}\n\n",
            i + 1,
            skill.name,
            skill.source,
            skill.content
        ));
    }

    let system_prompt = "你是一个 AI 技能评估专家。请分析以下技能,根据用户需求进行评分和描述。
重要:所有描述字段必须使用中文回答。

请按以下 JSON 格式返回分析结果(按相关性从高到低排序):
[
  {
    \"skill_id\": \"技能 ID\",
    \"skill_name\": \"技能名称(从 SKILL.md 的 name 字段提取)\",
    \"source\": \"来源仓库(如 vercel-labs/skills)\",
    \"score\": 9.2,
    \"description\": \"用中文描述这个技能是什么,能做什么\",
    \"how_to_use\": \"用中文简要说明使用方式\",
    \"reason\": \"用中文说明为什么推荐给这个用户\"
  }
]

评分标准:
- 9-10 分:完美匹配用户需求
- 7-8 分:高度相关
- 5-6 分:部分相关
- 1-4 分:关联度较低

要求:
- 尽可能推荐多个技能(至少 3-5 个,最多 10 个)
- 如果技能之间互补,可以同时推荐
- 每个技能都要有独特的推荐理由
- 只返回 JSON 数组,不要其他内容。description、how_to_use、reason 字段必须使用中文。";

    let user_prompt = format!(
        "用户需求:{}\n\n待分析技能:\n\n{}",
        user_query, skills_section
    );

    let base_url = if api_url.is_empty() {
        DEFAULT_AI_API_URL
    } else {
        api_url
    };

    let request_url = format!("{}/chat/completions", base_url);
    log::info!("AI batch analysis URL: {}", request_url);

    let request = ChatRequest {
        model: "MiniMax-M3".to_string(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_prompt,
            },
        ],
        temperature: 0.7,
        max_tokens: 4096,
    };

    let response = client
        .post(&request_url)
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .context("Failed to call AI API for batch analysis")?;

    let status = response.status();
    let body_text = response
        .text()
        .unwrap_or_else(|_| "<unreadable>".to_string());

    log::info!("AI batch analysis response status: {}", status);
    log::debug!("AI batch analysis response body: {}", body_text);

    if !status.is_success() {
        let error_detail = format!(
            "AI API error ({}): {}",
            status,
            truncate_error_body(&body_text)
        );
        return Err(anyhow::anyhow!(error_detail));
    }

    let resp: serde_json::Value = serde_json::from_str(&body_text)
        .context("Failed to parse AI API response")?;

    let content = extract_response_content(&resp);
    log::info!("AI batch analysis extracted content length: {}", content.len());

    let analyses = parse_ai_analysis_json(&content)?;
    Ok(analyses)
}

/// Parse AI analysis JSON response.
fn parse_ai_analysis_json(content: &str) -> Result<Vec<AiSkillAnalysis>> {
    let json_str = extract_json_from_content(content);

    // Parse as raw JSON first (AI may not return skill_name/source)
    #[derive(Debug, Deserialize)]
    struct RawAnalysis {
        skill_id: String,
        #[serde(default)]
        skill_name: String,
        #[serde(default)]
        source: String,
        score: f64,
        description: String,
        how_to_use: String,
        reason: String,
    }

    let raw: Vec<RawAnalysis> = serde_json::from_str(&json_str)
        .with_context(|| format!("Failed to parse AI analysis JSON: {}", json_str.chars().take(200).collect::<String>()))?;

    // Convert to AiSkillAnalysis, allowing empty skill_name/source to be filled later
    let analyses = raw.into_iter().map(|r| AiSkillAnalysis {
        skill_id: r.skill_id,
        skill_name: r.skill_name,
        source: r.source,
        score: r.score,
        description: r.description,
        how_to_use: r.how_to_use,
        reason: r.reason,
    }).collect();

    Ok(analyses)
}

/// Extract JSON array from content, handling markdown code blocks.
fn extract_json_from_content(content: &str) -> &str {
    if let Some(start) = content.find("```json") {
        let after_start = &content[start + 7..];
        if let Some(end) = after_start.find("```") {
            return after_start[..end].trim();
        }
    }

    if let Some(start) = content.find('[') {
        if let Some(end) = content.rfind(']') {
            return &content[start..=end];
        }
    }

    content
}
