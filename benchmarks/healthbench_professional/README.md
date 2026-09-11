# HealthBench Professional

Adds the `healthbench_professional` benchmark to Gym on top of the shared
`healthbench_rubric` resource server.

## Details

- Data source:
  [`openai/healthbench-professional`](https://huggingface.co/datasets/openai/healthbench-professional)
- Evaluation: rubric-based medical answer grading with an LLM judge
- Reward: HealthBench Professional length-adjusted rubric score
- Prompt: embedded in the prepared benchmark rows

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark healthbench_professional

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark healthbench_professional

# Collecting rollouts
gym eval run --no-serve \
    --agent healthbench_professional_benchmark_simple_agent \
    --input benchmarks/healthbench_professional/data/healthbench_professional_benchmark.jsonl \
    --output results/healthbench_professional/rollouts.jsonl \
    --num-repeats 6
```
