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


def test_config_loader_forwards_evolution_loop_sections() -> None:
    config = ConfigLoader().from_dict(
        {
            "pipeline": {
                "n_llm_workers": 3,
                "n_parents": 2,
                "output_mode": "diff",
            },
            "evaluator": {"parallel_evaluations": 5, "timeout": 45},
            "cascade": {"min_score_ratio": 0.8},
            "init": {"n_diverse_seeds": 0, "n_variants_per_seed": 0},
            "cvt": {"n_centroids": 8, "data_driven_centroids": True},
            "meta_advice": {"enabled": True, "interval": 24},
            "punctuated_equilibrium": {
                "interval": 12,
                "n_clusters": 1,
                "n_variants": 2,
            },
            "behavior": {"score_keys": ["combined_score", "validation_token_hit_rate"]},
        }
    )

    assert config.pipeline == {
        "n_llm_workers": 3,
        "n_parents": 2,
        "output_mode": "diff",
        "n_eval_processes": 5,
        "eval_timeout": 45,
    }
    assert config.init == {"n_diverse_seeds": 0, "n_variants_per_seed": 0}
    assert config.cvt == {"n_centroids": 8, "data_driven_centroids": True}
    assert config.meta_advice == {"enabled": True, "interval": 24}
    assert config.punctuated_equilibrium == {
        "interval": 12,
        "n_clusters": 1,
        "n_variants": 2,
    }
    assert config.cascade == {"min_score_ratio": 0.8}
    assert config.behavior["score_keys"] == [
        "combined_score",
        "validation_token_hit_rate",
    ]
    assert config.evolve_kwargs()["cvt"] == config.cvt
    assert config.evolve_kwargs()["meta_advice"] == config.meta_advice


def test_config_loader_forwards_evaluator_cascade_flag() -> None:
    config = ConfigLoader().from_dict({"evaluator": {"cascade_evaluation": False}})

    assert config.cascade == {"enabled": False}
    assert config.evolve_kwargs()["cascade"] == {"enabled": False}
