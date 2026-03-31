from pathlib import Path


def test_codex_skill_exists_and_mentions_server_validation() -> None:
    skill_path = Path(".codex/skills/vllm-plugin-version-adaptation/SKILL.md")
    assert skill_path.exists()

    text = skill_path.read_text()
    assert "api_server" in text
    assert "返回可读自然语言" in text
    assert "vllm 0.18.1" in text
    assert "upstream" in text
    assert "compat" in text
