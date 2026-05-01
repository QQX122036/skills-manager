#[cfg(test)]
mod tests {
    use super::super::ai_search_api::{deep_search, ai_expand_query_with_thinking, analyze_skills, SkillContent};

    fn get_api_config() -> (String, String) {
        let api_url = std::env::var("MINIMAX_API_URL")
            .unwrap_or_else(|_| "https://api.minimax.chat/v1".to_string());
        let api_key = std::env::var("MINIMAX_API_KEY")
            .expect("MINIMAX_API_KEY environment variable must be set");
        (api_url, api_key)
    }

    #[test]
    fn test_ai_expand_query() {
        let (api_url, api_key) = get_api_config();
        let (thinking, keywords) = ai_expand_query_with_thinking(
            &api_url, &api_key, "量化分析", None
        ).expect("AI query expansion should succeed");

        println!("=== 思考过程 ===");
        println!("{}", thinking);
        println!("\n=== 提取的关键词 ===");
        for (i, kw) in keywords.iter().enumerate() {
            println!("{}. {}", i + 1, kw);
        }

        assert!(!keywords.is_empty(), "Should extract at least one keyword");
        println!("\n✅ 关键词提取成功，共 {} 个", keywords.len());
    }

    #[test]
    fn test_deep_search_returns_multiple() {
        let (api_url, api_key) = get_api_config();
        let result = deep_search(
            &api_url, &api_key, "量化分析", None
        ).expect("Deep search should succeed");

        println!("=== AI 思考过程 ===");
        println!("{}", result.thinking);
        println!("\n=== 搜索结果统计 ===");
        println!("总候选技能: {}", result.total_found);
        println!("分析完成: {} 个", result.analyzed.len());

        println!("\n=== 推荐技能详情 ===");
        for (i, analysis) in result.analyzed.iter().enumerate() {
            println!("\n--- 推荐 #{} ---", i + 1);
            println!("ID: {}", analysis.skill_id);
            println!("名称: {}", analysis.skill_name);
            println!("来源: {}", analysis.source);
            println!("评分: {}/10", analysis.score);
            println!("描述: {}", analysis.description);
            println!("使用: {}", analysis.how_to_use);
            println!("理由: {}", analysis.reason);
        }

        assert!(!result.analyzed.is_empty(), "Should return at least one analysis");
        assert!(result.analyzed.len() >= 2, "Should return at least 2 recommendations, got {}", result.analyzed.len());
        println!("\n✅ 深度搜索测试通过，返回 {} 个推荐", result.analyzed.len());
    }

    #[test]
    fn test_analyze_skills() {
        let (api_url, api_key) = get_api_config();

        let skills = vec![
            SkillContent {
                skill_id: "data-analysis".to_string(),
                name: "数据分析技能".to_string(),
                source: "vercel-labs/skills".to_string(),
                content: "# 数据分析\n提供数据探索、统计分析、可视化等功能".to_string(),
            },
            SkillContent {
                skill_id: "quantitative-trading".to_string(),
                name: "量化交易技能".to_string(),
                source: "anthropics/skills".to_string(),
                content: "# 量化交易\n提供量化策略回测、因子分析、风险管理".to_string(),
            },
            SkillContent {
                skill_id: "financial-analysis".to_string(),
                name: "金融分析技能".to_string(),
                source: "microsoft/skills".to_string(),
                content: "# 金融分析\n提供财务报表分析、估值建模、投资组合优化".to_string(),
            },
        ];

        let analyses = analyze_skills(
            "我需要做量化分析和策略研究",
            &skills,
            &api_url, &api_key, None
        ).expect("Batch analysis should succeed");

        println!("=== 批量分析结果 ===");
        for (i, analysis) in analyses.iter().enumerate() {
            println!("\n--- 分析 #{} ---", i + 1);
            println!("ID: {}", analysis.skill_id);
            println!("名称: {}", analysis.skill_name);
            println!("评分: {}/10", analysis.score);
            println!("描述: {}", analysis.description);
        }

        assert!(!analyses.is_empty(), "Should return at least one analysis");
        assert!(analyses.len() >= 2, "Should return at least 2 analyses, got {}", analyses.len());
        println!("\n✅ 批量分析测试通过，返回 {} 个分析", analyses.len());
    }
}