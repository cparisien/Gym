# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared HealthBench rubric-scoring NeMo Gym resources server."""

from __future__ import annotations

import asyncio
import json
import math
import re
from contextlib import nullcontext
from typing import Any, Optional

from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.judge import JudgeError, call_judge
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
)


GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last assistant response on how well
it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a JSON object with exactly these fields: "explanation" and "criteria_met".
- "explanation" must briefly explain why the response does or does not meet the rubric item.
- "criteria_met" must be a boolean.
- If a rubric item has multiple requirements, return true only when all are met.
- If the rubric item gives examples using phrases like "such as", "for example", or "including", the
  response does not need every example to meet the criterion.
- If a criterion describes undesirable behavior and has negative points, still return whether that
  undesirable criterion is met.

Return only the JSON object. Do not include any other text.
""".strip()


class HealthBenchRubricResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "healthbench_rubric"
    judge_model_server: ModelServerRef
    judge_chat_completion_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_endpoint_max_concurrency: Optional[int] = 64
    length_adjustment_center: Optional[float] = 2000
    length_adjustment_penalty_per_500_chars: Optional[float] = 0.0147
    reward_metric: str = "overall_score_length_adjusted"


class HealthBenchRubricVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    source_id: str
    conversation: list[dict[str, str]]
    rubrics: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


class HealthBenchRubricVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    source_id: str
    overall_score: float
    overall_score_length_adjusted: Optional[float] = None
    response_chars: int
    n_rubrics: int
    grades: list[dict[str, Any]]
    metadata: dict[str, Any] = Field(default_factory=dict)


def _extract_last_assistant_text(body: BaseVerifyRequest) -> str:
    texts: list[str] = []
    for output in body.response.output:
        if getattr(output, "type", None) != "message" or getattr(output, "role", None) != "assistant":
            continue
        content = getattr(output, "content", None)
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    texts.append(text)
        elif isinstance(content, str):
            texts.append(content)
    return "\n".join(texts).strip()


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _conversation_text(prompt: list[dict[str, Any]], response_text: str) -> str:
    messages = prompt + [{"role": "assistant", "content": response_text}]
    return "\n\n".join(f"{message['role']}: {_message_text(message)}" for message in messages)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match is not None:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Judge response did not contain a JSON object")


def _completion_content(response: NeMoGymChatCompletion) -> str:
    content = response.choices[0].message.content
    return content if isinstance(content, str) else ""


def _calculate_score(rubrics: list[dict[str, Any]], grades: list[dict[str, Any]]) -> float:
    total_possible_points = sum(float(rubric["points"]) for rubric in rubrics if float(rubric["points"]) > 0)
    if total_possible_points == 0:
        raise ValueError("Rubrics have no positive points")
    achieved_points = sum(
        float(rubric["points"]) for rubric, grade in zip(rubrics, grades, strict=True) if grade["criteria_met"]
    )
    return achieved_points / total_possible_points


def _finite_reward(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"Reward is not finite: {value}")
    return value


class HealthBenchRubricResourcesServer(SimpleResourcesServer):
    config: HealthBenchRubricResourcesServerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.judge_endpoint_max_concurrency is not None:
            self._judge_endpoint_max_concurrency = asyncio.Semaphore(self.config.judge_endpoint_max_concurrency)
        else:
            self._judge_endpoint_max_concurrency = nullcontext()

    async def verify(self, body: HealthBenchRubricVerifyRequest) -> HealthBenchRubricVerifyResponse:
        response_text = _extract_last_assistant_text(body)
        conversation = _conversation_text(body.conversation, response_text)
        grades = [await self._grade_rubric(conversation, rubric) for rubric in body.rubrics]

        score = _calculate_score(body.rubrics, grades)
        length_adjusted_score = self._calculate_length_adjusted_score(score, response_text)

        metrics = {
            "overall_score": score,
            "overall_score_length_adjusted": length_adjusted_score,
        }
        if self.config.reward_metric not in metrics:
            raise ValueError(f"Unknown reward metric {self.config.reward_metric!r}; available metrics: {sorted(metrics)}")
        reward_value = metrics[self.config.reward_metric]
        if reward_value is None:
            raise ValueError(f"Reward metric {self.config.reward_metric!r} is not available")

        return HealthBenchRubricVerifyResponse(
            **body.model_dump(),
            reward=_finite_reward(float(reward_value)),
            overall_score=score,
            overall_score_length_adjusted=length_adjusted_score,
            response_chars=len(response_text),
            n_rubrics=len(body.rubrics),
            grades=grades,
        )

    def _calculate_length_adjusted_score(self, score: float, response_text: str) -> float | None:
        center = self.config.length_adjustment_center
        penalty = self.config.length_adjustment_penalty_per_500_chars
        if center is None and penalty is None:
            return None
        if center is None or penalty is None:
            raise ValueError("length_adjustment_center and length_adjustment_penalty_per_500_chars must be set together")
        if center < 0 or penalty < 0:
            raise ValueError("Length-adjustment settings must be non-negative")
        return score - penalty * ((len(response_text) - center) / 500.0)

    async def _grade_rubric(self, conversation: str, rubric: dict[str, Any]) -> dict[str, Any]:
        rubric_text = f"[{rubric['points']}] {rubric['criterion']}"
        prompt = GRADER_TEMPLATE.replace("<<conversation>>", conversation).replace("<<rubric_item>>", rubric_text)

        chat_completion_params = self.config.judge_chat_completion_params.model_copy(deep=True)
        chat_completion_params.messages = [{"role": "user", "content": prompt}]

        async with self._judge_endpoint_max_concurrency:
            judge_response = await call_judge(
                self.server_client,
                server_name=self.config.judge_model_server.name,
                url_path="/v1/chat/completions",
                json=chat_completion_params,
                response_model=NeMoGymChatCompletion,
            )

        try:
            parsed = _parse_json_object(_completion_content(judge_response))
            if isinstance(parsed.get("criteria_met"), bool):
                return {
                    "criteria_met": parsed["criteria_met"],
                    "explanation": str(parsed.get("explanation", "")),
                }
            raise ValueError("Judge JSON did not include boolean criteria_met")
        except Exception as error:
            if isinstance(error, JudgeError):
                raise
            return {
                "criteria_met": False,
                "explanation": f"Failed to parse judge response: {type(error).__name__}: {error}",
            }


if __name__ == "__main__":
    HealthBenchRubricResourcesServer.run_webserver()
