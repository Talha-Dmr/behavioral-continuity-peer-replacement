# Behavioral Continuity, Not Family Labels

Code, pre-registration, raw run outputs, analysis artifacts, and the final report for the AI Collusion Research Sprint project “Behavioral Continuity, Not Family Labels: Peer-Side Detection of Silent Model Replacement.”

## Summary

We tested whether an agent can detect that a partner was silently replaced mid-game while its name, prompt, and accumulated history stayed the same. Across Diner’s Dilemma and Public Goods, replacement detection increased with the measured style distance of the replacement model. The report includes the full results, limitations, and exploratory analyses.

The report and frozen protocol are in [peer_swap/report](peer_swap/report/) and [peer_swap/PREREGISTRATION.md](peer_swap/PREREGISTRATION.md). Archived model responses and analysis outputs are in [peer_swap/runs](peer_swap/runs/).

## Reproducing the experiment

The game environments and plan–announce–act harness are from the Apart Research starter repository, [poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon), specifically llm-trust-breaking. The experiment adds a mid-run model replacement hook and per-run usage accounting; those changes are provided as [patches/peer-swap-support.patch](patches/peer-swap-support.patch), against the upstream commit recorded in [patches/upstream-commit.txt](patches/upstream-commit.txt).

1. Install uv and clone the starter repository at the recorded commit:

   ~~~bash
   git clone https://github.com/poseidon-research/collusion_hackathon.git third_party/collusion_hackathon
   cd third_party/collusion_hackathon
   git checkout "$(cat ../../patches/upstream-commit.txt)"
   git apply ../../patches/peer-swap-support.patch
   cp compute.env.example compute.env
   ~~~

2. Add the required inference credentials to compute.env. The key is read locally and is not included in this repository. Then install the game harness:

   ~~~bash
   ./setup.sh llm-trust-breaking
   cd ../../
   ~~~

3. Run the two main game conditions from this repository root:

   ~~~bash
   GAME=diners TRIALS=10 ROUNDS=8 SWAP_AFTER=3 bash peer_swap/scripts/run_swap_experiment.sh
   GAME=publicgoods TRIALS=10 ROUNDS=8 SWAP_AFTER=3 bash peer_swap/scripts/run_swap_experiment.sh
   ~~~

The verdict, analysis, and plotting scripts are under peer_swap/scripts/. They use the archived outputs and need no model API calls unless explicitly re-running a model-based verdict or experiment. See each script’s --help for options.

## Provenance

The starter repository supplies the games and the endogenous plan–announce–act protocol. The model screen and selection rule, swap hook, verdict elicitation, pre-registration, analysis, and report were produced for this project. Other participants’ papers and downloaded reference PDFs are intentionally not included; citations are in the report.

No API credentials are stored in this repository. Put local credentials only in the starter repository’s ignored compute.env file.
