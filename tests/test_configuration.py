"""Tests for Levi workflow configuration loading."""

from randomize_evolve.workflow.configuration import ConfigLoader


def test_config_loader_includes_prompt_sections_in_problem_description() -> None:
    config = ConfigLoader().from_dict(
        {
            "max_iterations": 3,
            "problem": {"description": "Base problem."},
            "prompt": {
                "system_message": "Use block.prefix_hash as the stable key.",
                "objectives": ["Avoid id(block)."],
            },
            "search": {"notes": "Do not guess schema fields."},
        }
    )

    assert "Base problem." in config.problem_description
    assert "Use block.prefix_hash as the stable key." in config.problem_description
    assert "- Avoid id(block)." in config.problem_description
    assert "Do not guess schema fields." in config.problem_description
