import os
from typing import Any, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

tokenizer = AutoTokenizer.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
model = AutoModelForCausalLM.from_pretrained("StabilityAI/stablelm-tuned-alpha-7b")
model.half().cuda()


class ChatEntry:
    category: str
    message: str

    def __init__(self, category: str, message: str) -> None:
        self.category = category.upper()
        self.message = message.strip()

    def __str__(self) -> str:
        msg = self.message.strip()
        if not msg.endswith("."):
            msg += "."
        return f"<|{self.category.upper()}|>{msg}\n"


class ChatLog:
    entries: List[ChatEntry]

    def __init__(self, text: str) -> None:
        parts = [i.strip() for i in text.split("<|")]
        parts = [i.split("|>") for i in parts if i]
        for i in parts:
            if len(i) != 2:
                print(f"Warn: found text entry in chat with {len(i)} parts:{i}")
        self.entries = [
            ChatEntry(*i) for i in parts if len(i) == 2 and i[1] and i[0] != "ENDOFTEXT"
        ]

    def __str__(self) -> str:
        return "".join(e.__str__() for e in self.entries)


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class ChatAgent:
    def __init__(self, theme: str) -> None:
        self.theme = "fantasy"
        self.system_prompt = f"""<|SYSTEM|># StableLM Adventure Story Generator
            - StableLM acts like a {theme} story writer.
            - StableLM also acts like a Dungeons and Dragons DM.
            - StableLM creates interesting and engaging stories.
            - StableLM continues the narrative by responding to user decisions.
            - StableLM clearly describes the scene for the user.
            """
        req = (
            f"Describe the introduction to a {theme} choose your own adventure story"
            " with a clear setting and call to action for the user."
        )
        prompt = f"{self.system_prompt}<|USER|>{req}"
        self.chat_log = ChatLog(prompt)
        self._sample()

    def _sample(self) -> ChatLog:
        prompt = f"{self.chat_log}<|ASSISTANT|>"
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
        )
        self.chat_log = ChatLog(tokenizer.decode(tokens[0]))
        return self.chat_log

    def input(self, req: str) -> ChatLog:
        self.chat_log.entries.append(ChatEntry("USER", req))
        return self._sample()
