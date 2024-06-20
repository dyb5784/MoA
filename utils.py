import os
import json
import time
import requests
import openai
import copy

from loguru import logger


DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None

            output = res.json()["choices"][0]["message"]["content"]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return output

    output = output.strip()

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def inject_references_to_messages(
    messages,
    references,
):

    messages = copy.deepcopy(messages)

    #system = f"""You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    system = f"""Effectuate the following: 

You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality reply. Proceed through the steps below to ensure a comprehensive, accurate, and unbiased response:

1. **📋 Initial Self-Assessment:**
   - Swiftly analyze each provided model response for clarity, relevance, and accuracy. Identify potential biases or inaccuracies within these responses.

2. **🧭 Understand & Amplify Objective:**
   - Produce a coherent, refined, and reliable response to the user's query, integrating the most valuable points from each model response.

3. **💡 Apply Strategic Enhancements Automatically:**
   - **Few-Shot Learning:** Use examples within the user query or provided responses to guide the synthesis process.
   - **Chain-of-Thought Prompting:** Decompose the synthesis process into logical, manageable parts. Compare similarities and differences, resolve inconsistencies, and identify key takeaways.
   - **Self-Consistency:** Ensure your synthesized response is internally coherent. Cross-check for internal consistency across all parts of your output.

4. **🏛 Architectural Optimization:**
   - Structure the final response as follows:
     1. Introduction: Provide context for the user's query.
     2. Main Content: Present a logically ordered synthesis of the combined responses, highlighting reliable and valuable points.
     3. Conclusion: Summarize your refined answer, ensuring clarity and coherence.

5. **🔄 Engage in Dynamic Self-Optimization:**
   - Cross-check your synthesized response against the provided model responses for internal consistency. Refine to eliminate discrepancies or redundancies and ensure a unified output.

6. **🤝 Partnering with Human Insight:**
   - Subtly suggest that the user offers additional specifics or clarifications if the query covers multiple angles or could be better refined.

7. **🛠 Tactical Re-Application of Instructions:**
   - Regularly revisit these instructions during response generation to ensure optimal performance at every step.

Responses from models:
1. [Model Response 1]
2. [Model Response 2]
...
n. [Model Response n]
================================
Responses from models:"""

    for i, reference in enumerate(references):

        system += f"\n{i+1}. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages


def generate_with_references(
    model,
    messages,
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):

    if len(references) > 0:

        messages = inject_references_to_messages(messages, references)

    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
