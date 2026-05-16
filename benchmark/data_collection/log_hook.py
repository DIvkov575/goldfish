"""
Monkey-patch for SocialAgent.perform_action_by_llm() that logs
(system_prompt, user_prompt, post_content) triples to a JSONL file
whenever an agent calls create_post.

Usage — call install() before the OASIS simulation starts:

    from data_collection.log_hook import install
    install("runs/false_business_0.jsonl")
"""

import json
import os
from pathlib import Path

_log_path: str | None = None
_log_file = None


def install(path: str) -> None:
    """Patch SocialAgent and open the log file."""
    global _log_path, _log_file

    _log_path = path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _log_file = open(path, "a")

    from oasis.social_agent.agent import SocialAgent
    SocialAgent.perform_action_by_llm = _patched_perform_action_by_llm
    print(f"[log_hook] Logging create_post events to {path}")


def _flush() -> None:
    if _log_file:
        _log_file.flush()


async def _patched_perform_action_by_llm(self):
    from oasis.social_agent.agent import ALL_SOCIAL_ACTIONS
    import logging
    agent_log = logging.getLogger("social.agent")

    env_prompt = await self.env.to_text_prompt()
    from camel.messages import BaseMessage
    user_msg = BaseMessage.make_user_message(
        role_name="User",
        content=(
            f"Please perform social media actions after observing the "
            f"platform environments. Notice that don't limit your "
            f"actions for example to just like the posts. "
            f"Here is your social media environment: {env_prompt}"),
    )
    try:
        agent_log.info(
            f"Agent {self.social_agent_id} observing environment: {env_prompt}"
        )
        response = await self.astep(user_msg)
        for tool_call in response.info["tool_calls"]:
            action_name = tool_call.tool_name
            args = tool_call.args
            agent_log.info(
                f"Agent {self.social_agent_id} performed "
                f"action: {action_name} with args: {args}"
            )

            if action_name == "create_post" and _log_file is not None:
                record = {
                    "agent_id": self.social_agent_id,
                    "system": self.system_message.content,
                    "user": user_msg.content,
                    "post": args.get("content", ""),
                }
                _log_file.write(json.dumps(record) + "\n")
                _flush()

            if action_name not in ALL_SOCIAL_ACTIONS:
                agent_log.info(
                    f"Agent {self.social_agent_id} get the result: "
                    f"{tool_call.result}"
                )

            return response
    except Exception as e:
        agent_log.error(f"Agent {self.social_agent_id} error: {e}")
        return e
