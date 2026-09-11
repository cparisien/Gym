# HealthBench

Adds the `healthbench` benchmark to Gym on top of the shared
`healthbench_rubric` resource server.

## Details

- Data source:
  [`openai/healthbench`](https://huggingface.co/datasets/openai/healthbench)
- Default subset: `oss_eval`
- Evaluation: rubric-based medical answer grading with an LLM judge
- Reward: HealthBench overall rubric score
- Prompt: embedded in the prepared benchmark rows

## Example usage

```bash
# Prepare benchmark data
gym eval prepare --benchmark healthbench

# Running servers
gym env start \
    --model-type vllm_model \
    --benchmark healthbench

# Collecting rollouts
gym eval run --no-serve \
    --agent healthbench_benchmark_simple_agent \
    --input benchmarks/healthbench/data/healthbench_benchmark.jsonl \
    --output results/healthbench/rollouts.jsonl
```
