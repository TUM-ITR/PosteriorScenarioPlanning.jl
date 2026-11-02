using Documenter, PosteriorScenarioPlanning

# Local builds (outside CI) use plain file links so that the pages can be opened
# directly in a browser. The repository is set explicitly, so that source links
# work regardless of the git remote of the local checkout.
const IS_CI = get(ENV, "CI", nothing) == "true"

makedocs(
    sitename="PosteriorScenarioPlanning.jl",
    modules=[PosteriorScenarioPlanning],
    checkdocs=:exports,
    warnonly=[:missing_docs, :cross_references],
    repo=Remotes.GitHub("TUM-ITR", "PosteriorScenarioPlanning.jl"),
    format=Documenter.HTML(;
        prettyurls=IS_CI,
        edit_link="main",
        mathengine=Documenter.MathJax3(),
        size_threshold_warn=200 * 2^10,
    ),
    pages=[
        "Home" => "index.md",
        "Method" => [
            "Overview" => "method/overview.md",
            "Posterior Inference" => "method/inference.md",
            "Validation and Certification" => "method/validation.md",
            "Synthesis and Calibration" => "method/synthesis.md",
            "Receding-Horizon Planning" => "method/receding_horizon.md",
        ],
        "Choosing the Parameters" => "parameters.md",
        "Case Study: Vertical Farm" => [
            "Overview" => "case_study/overview.md",
            "Crop Model and Prior" => "case_study/model_and_prior.md",
            "Control Problem" => "case_study/control.md",
            "Sampler Tuning" => "case_study/sampler_tuning.md",
            "Calibration" => "case_study/calibration.md",
            "Open-Loop Results" => "case_study/open_loop.md",
            "Receding-Horizon Results" => "case_study/receding_horizon.md",
        ],
        "API Reference" => "api.md",
    ],
)

if IS_CI
    deploydocs(
        repo="github.com/TUM-ITR/PosteriorScenarioPlanning.jl.git",
    )
end
