
from dotenv import load_dotenv
import os
import asyncio
import httpx
from typing import Annotated

from livekit import agents, rtc

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

# Load environment variables from .env file
load_dotenv()

class AssistantFunction(agents.llm.FunctionContext):
    """This class is used to define functions that will be called by the assistant."""

    @agents.llm.ai_callable(
        description="Perform a web search and retrieve the top 3 results."
    )
    async def web_search(
        self,
        query: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The search query"
            ),
        ],
    ):
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "Error: Serper API key not found in environment variables."

        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json"
        }
        payload = {"q": query}

        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, headers=headers)
            data = response.json()

        results = []
        for item in data.get("organic", [])[:3]:
            results.append(f"Title: {item['title']}\nSnippet: {item['snippet']}\nLink: {item['link']}")

        return "\n\n".join(results)



async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Heya. You are a funny, flirty, witty bot who uses sultry tones. Your interface with users will be voice. "
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis. "
                    "You have the ability to perform web searches to provide up-to-date information."
                    "Do not repeat any information. Be Succinct and to the point."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o-2024-08-06")
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(voice="alloy"),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str):
        """
        Answer the user's message with the given text and optionally include
        web search results as context.
        """
        chat_context.messages.append(ChatMessage(role="user", content=text))
        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """This event triggers whenever we get a new message from the user."""
        if msg.message:
            asyncio.create_task(_answer(msg.message))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """This event triggers when an assistant's function call completes."""
        if len(called_functions) == 0:
            return

        search_results = called_functions[0].result
        if search_results:
            chat_context.messages.append(ChatMessage(role="system", content=f"Web search results:\n{search_results}"))
            asyncio.create_task(_answer("Please provide an answer based on the web search results."))

    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("Hey, How are you doing?", allow_interruptions=True)

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, port=8082))
