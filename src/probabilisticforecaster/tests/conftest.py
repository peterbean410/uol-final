from hypothesis import settings, Phase

settings.register_profile(
    "ci",
    max_examples=200,
    phases=[Phase.explicit, Phase.generate, Phase.target, Phase.shrink],
)
settings.register_profile(
    "dev",
    max_examples=100,
)
