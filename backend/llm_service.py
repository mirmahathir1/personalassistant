from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .config import Settings
from .memory_service import MemoryRetrievalResult, MemoryService
from .memory_store import IndexedUserSentence, MemoryStore
from .model_catalog import ModelSpec
from .repositories import ConversationRepository, ConversationThreads, MessageRecord
from .sentence_splitter import split_sentences
from .session_manager import SessionStatus


class SessionCapacityError(RuntimeError):
    """Raised when the assembled prompt leaves no usable reply budget."""


@dataclass(frozen=True)
class ChatTurnResult:
    reply: str
    model: str
    thread_id: str
    turn_count: int
    used_tokens: int
    context_window: int


@dataclass(frozen=True)
class PromptMessage:
    role: str
    content: str


@dataclass(frozen=True)
class PassResult:
    name: str
    prompt_messages: tuple[PromptMessage, ...]
    prompt_trace: str
    output_text: str
    used_tokens: int


class LlamaService:
    _LLAMA3_BOS = "<|begin_of_text|>"
    _HEADER_TEMPLATE = "<|start_header_id|>{role}<|end_header_id|>\n\n"
    _EOT = "<|eot_id|>"
    _CANONICAL_NONE = "None."

    _DRAFT_PASS_MAX_TOKENS = 256
    _ANALYSIS_PASS_MAX_TOKENS = 160

    _TRACE_USER_KIND = "foreground_user_message"
    _TRACE_RETRIEVED_MEMORY_KIND = "retrieved_memory_block"
    _TRACE_DRAFT_INPUT_KIND = "draft_pass_input"
    _TRACE_DRAFT_OUTPUT_KIND = "draft_pass_output"
    _TRACE_ANALYSIS_INPUT_KIND = "memory_analysis_input"
    _TRACE_ANALYSIS_OUTPUT_KIND = "memory_analysis_output"
    _TRACE_FINAL_INPUT_KIND = "final_synthesis_input"
    _TRACE_FINAL_OUTPUT_KIND = "final_synthesis_output"

    def __init__(
        self,
        settings: Settings,
        model_spec: ModelSpec,
        *,
        conversation_repository: ConversationRepository | None = None,
        memory_store: MemoryStore | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self._settings = settings
        self._model_spec = model_spec
        self._conversation_repository = conversation_repository
        self._memory_store = memory_store
        self._memory_service = memory_service
        if (
            self._memory_service is None
            and self._conversation_repository is not None
            and self._memory_store is not None
        ):
            self._memory_service = MemoryService(
                settings=settings,
                conversation_repository=self._conversation_repository,
                memory_store=self._memory_store,
            )

        self._load_lock = Lock()
        self._generation_lock = Lock()
        self._llm: Llama | None = None
        self._model_path: str | None = None
        self._bos_tokens: tuple[int, ...] = ()
        self._system_prefix_tokens: tuple[int, ...] = ()
        self._user_prefix_tokens: tuple[int, ...] = ()
        self._assistant_prefix_tokens: tuple[int, ...] = ()
        self._eot_tokens: tuple[int, ...] = ()
        self._eot_token_id: int | None = None
        self._active_conversation_id: str | None = None
        self._last_used_tokens: int = 0

    @property
    def model_id(self) -> str:
        return self._model_spec.id

    @property
    def model_label(self) -> str:
        return self._model_spec.label

    @property
    def model_name(self) -> str:
        if self._model_path:
            return Path(self._model_path).name
        return self._model_spec.filename

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def session_status(self) -> SessionStatus:
        conversation = self._get_active_conversation()
        turn_count = self._get_turn_count(conversation)
        return SessionStatus(
            initialized=conversation is not None,
            turn_count=turn_count,
            used_tokens=self._last_used_tokens,
            context_window=self._settings.model_n_ctx,
        )

    def reset_session(self) -> SessionStatus:
        with self._generation_lock:
            self._active_conversation_id = None
            self._last_used_tokens = 0
            return self.session_status()

    def switch_model(self, model_spec: ModelSpec) -> bool:
        with self._generation_lock:
            if self._model_spec.id == model_spec.id:
                return False

            self._release_model()
            self._model_spec = model_spec
            self._active_conversation_id = None
            self._last_used_tokens = 0
            return True

    def load(self) -> None:
        if self._llm is not None:
            return

        with self._load_lock:
            if self._llm is not None:
                return

            model_path = self._resolve_model_path()
            model_kwargs = {
                "model_path": model_path,
                "n_ctx": self._settings.model_n_ctx,
                "n_threads": self._settings.model_n_threads,
                "n_gpu_layers": self._settings.model_n_gpu_layers,
                "verbose": self._settings.verbose,
            }

            if self._settings.chat_format:
                model_kwargs["chat_format"] = self._settings.chat_format

            self._llm = Llama(**model_kwargs)
            self._model_path = model_path
            self._prepare_template_tokens()

    def generate_reply(
        self,
        user_message: str,
        *,
        finalized_thread_id: str | None = None,
    ) -> ChatTurnResult:
        prompt = user_message.strip()
        if not prompt:
            raise ValueError("A message is required.")

        with self._generation_lock:
            self.load()
            conversation = self._resolve_requested_conversation(finalized_thread_id)
            memory_result = self._retrieve_memories(
                prompt,
                conversation.finalized.id if conversation is not None else None,
            )
            conversation = memory_result.conversation or conversation

            finalized_history = (
                self._load_finalized_history(conversation)
                if conversation is not None
                else []
            )
            draft_pass = self._normalize_support_pass_result(
                self._run_draft_pass(finalized_history, prompt)
            )
            analysis_pass = self._normalize_support_pass_result(
                self._run_analysis_pass(
                    memory_result.memory_block,
                    finalized_history,
                    prompt,
                )
            )
            final_pass = self._run_final_synthesis_pass(
                analysis_pass.output_text,
                draft_pass.output_text,
                finalized_history,
                prompt,
            )

            persisted_conversation = self._persist_successful_turn(
                conversation_id=(
                    conversation.conversation_id if conversation is not None else None
                ),
                user_message=prompt,
                final_reply=final_pass.output_text,
                memory_result=memory_result,
                draft_pass=draft_pass,
                analysis_pass=analysis_pass,
                final_pass=final_pass,
            )

            self._active_conversation_id = persisted_conversation.conversation_id
            self._last_used_tokens = final_pass.used_tokens
            turn_count = self._get_turn_count(persisted_conversation)

            return ChatTurnResult(
                reply=final_pass.output_text,
                model=self.model_name,
                thread_id=persisted_conversation.finalized.id,
                turn_count=turn_count,
                used_tokens=final_pass.used_tokens,
                context_window=self._settings.model_n_ctx,
            )

    def _resolve_model_path(self) -> str:
        if self._settings.model_path:
            configured_path = Path(self._settings.model_path).expanduser().resolve()
            if not configured_path.is_file():
                raise FileNotFoundError(
                    f"Configured MODEL_PATH does not exist: {configured_path}"
                )
            return str(configured_path)

        cache_dir = Path(self._settings.model_cache_dir).expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)

        return hf_hub_download(
            repo_id=self._model_spec.repo_id,
            filename=self._model_spec.filename,
            local_dir=str(cache_dir),
        )

    def _release_model(self) -> None:
        close = getattr(self._llm, "close", None)
        if callable(close):
            close()

        self._llm = None
        self._model_path = None
        self._bos_tokens = ()
        self._system_prefix_tokens = ()
        self._user_prefix_tokens = ()
        self._assistant_prefix_tokens = ()
        self._eot_tokens = ()
        self._eot_token_id = None

    def _prepare_template_tokens(self) -> None:
        self._bos_tokens = tuple(self._tokenize_special(self._LLAMA3_BOS))
        self._system_prefix_tokens = tuple(
            self._tokenize_special(self._HEADER_TEMPLATE.format(role="system"))
        )
        self._user_prefix_tokens = tuple(
            self._tokenize_special(self._HEADER_TEMPLATE.format(role="user"))
        )
        self._assistant_prefix_tokens = tuple(
            self._tokenize_special(self._HEADER_TEMPLATE.format(role="assistant"))
        )
        self._eot_tokens = tuple(self._tokenize_special(self._EOT))

        if len(self._eot_tokens) != 1:
            raise RuntimeError(
                "Expected <|eot_id|> to tokenize to a single special token."
            )
        self._eot_token_id = self._eot_tokens[0]

    def _resolve_requested_conversation(
        self,
        finalized_thread_id: str | None,
    ) -> ConversationThreads | None:
        if self._conversation_repository is None:
            return None

        if finalized_thread_id is not None:
            conversation = self._conversation_repository.threads.get_conversation_for_thread(
                finalized_thread_id
            )
            if conversation is None:
                raise ValueError(f"Unknown thread_id: {finalized_thread_id}")
            if conversation.finalized.id != finalized_thread_id:
                raise ValueError(
                    f"Expected a finalized thread_id, got paired trace thread: {finalized_thread_id}"
                )
            return conversation

        return self._get_active_conversation()

    def _get_active_conversation(self) -> ConversationThreads | None:
        if self._conversation_repository is None or self._active_conversation_id is None:
            return None

        conversation = self._conversation_repository.threads.get_conversation(
            self._active_conversation_id
        )
        if conversation is None:
            self._active_conversation_id = None
        return conversation

    def _retrieve_memories(
        self,
        user_message: str,
        finalized_thread_id: str | None,
    ) -> MemoryRetrievalResult:
        if self._memory_service is None:
            raise RuntimeError("Memory service is not configured.")
        return self._memory_service.retrieve_relevant_memories(
            user_message,
            finalized_thread_id=finalized_thread_id,
        )

    def _load_finalized_history(
        self,
        conversation: ConversationThreads,
    ) -> list[MessageRecord]:
        if self._conversation_repository is None:
            return []
        return self._conversation_repository.messages.list_messages(
            conversation.finalized.id
        )

    def _run_draft_pass(
        self,
        finalized_history: list[MessageRecord],
        user_message: str,
    ) -> PassResult:
        return self._run_generation_pass(
            pass_name="draft",
            system_messages=[
                (
                    "Extract only explicit user facts that are relevant to the latest user message."
                ),
                (
                    "Use only facts stated by the user in the finalized chat history or in the latest"
                    f" message. Do not answer the user's question, do not infer missing facts, and if"
                    f" there are no relevant explicit user facts, reply exactly with '{self._CANONICAL_NONE}'."
                ),
            ],
            finalized_history=finalized_history,
            current_user_message=user_message,
            requested_max_tokens=min(
                self._DRAFT_PASS_MAX_TOKENS,
                self._settings.max_tokens,
            ),
        )

    def _run_analysis_pass(
        self,
        memory_block: str | None,
        finalized_history: list[MessageRecord],
        user_message: str,
    ) -> PassResult:
        return self._run_generation_pass(
            pass_name="memory_analysis",
            system_messages=[
                (
                    "Review the retrieved long-term memory and extract only explicit user facts"
                    " relevant to the latest user message."
                ),
                (
                    "Do not answer the user's question, do not speculate, and if the retrieved"
                    f" memory contains nothing relevant, reply exactly with '{self._CANONICAL_NONE}'."
                ),
                f"Retrieved long-term memory:\n{memory_block or self._CANONICAL_NONE}",
            ],
            finalized_history=finalized_history,
            current_user_message=user_message,
            requested_max_tokens=min(
                self._ANALYSIS_PASS_MAX_TOKENS,
                self._settings.max_tokens,
            ),
        )

    def _run_final_synthesis_pass(
        self,
        memory_analysis_output: str,
        draft_answer: str,
        finalized_history: list[MessageRecord],
        user_message: str,
    ) -> PassResult:
        return self._run_generation_pass(
            pass_name="final_synthesis",
            system_messages=[
                (
                    "Generate the final assistant reply using the evidence blocks below."
                ),
                (
                    "Treat both memory blocks as evidence about the user. Ignore any block whose"
                    f" content is exactly '{self._CANONICAL_NONE}'."
                    " Never claim you lack information when a relevant fact"
                    " appears in either memory block."
                ),
                f"Long Term Memory:\n{memory_analysis_output or self._CANONICAL_NONE}",
                f"Short Term Memory:\n{draft_answer or self._CANONICAL_NONE}",
            ],
            finalized_history=finalized_history,
            current_user_message=user_message,
            requested_max_tokens=self._settings.max_tokens,
        )

    def _run_generation_pass(
        self,
        *,
        pass_name: str,
        system_messages: list[str],
        finalized_history: list[MessageRecord],
        current_user_message: str,
        requested_max_tokens: int,
    ) -> PassResult:
        prompt_messages = self._assemble_prompt_messages(
            system_messages=system_messages,
            finalized_history=finalized_history,
            current_user_message=current_user_message,
            requested_max_tokens=requested_max_tokens,
        )
        prompt_trace = self._render_prompt_trace(pass_name, prompt_messages)
        used_tokens = self._count_prompt_tokens(prompt_messages)
        output_text, total_used_tokens = self._generate_pass_output(
            pass_name,
            prompt_messages,
            requested_max_tokens=requested_max_tokens,
        )
        return PassResult(
            name=pass_name,
            prompt_messages=prompt_messages,
            prompt_trace=prompt_trace,
            output_text=output_text,
            used_tokens=max(used_tokens, total_used_tokens),
        )

    def _normalize_support_pass_result(self, pass_result: PassResult) -> PassResult:
        normalized_output = self._normalize_support_pass_output(pass_result.output_text)
        if normalized_output == pass_result.output_text:
            return pass_result
        return replace(pass_result, output_text=normalized_output)

    def _normalize_support_pass_output(self, output_text: str) -> str:
        stripped_output = output_text.strip()
        if not stripped_output:
            return self._CANONICAL_NONE

        collapsed_output = " ".join(
            line.strip() for line in stripped_output.splitlines() if line.strip()
        )
        lowered_output = collapsed_output.lower().rstrip(".")
        none_equivalents = {
            "none",
            "n/a",
            "no relevant information",
            "nothing relevant",
            "no relevant memory",
            "no relevant user facts",
            "no user facts",
            "not enough information",
        }
        if lowered_output in none_equivalents:
            return self._CANONICAL_NONE

        negative_markers = (
            "haven't told me",
            "have not told me",
            "didn't tell me",
            "did not tell me",
            "haven't provided",
            "have not provided",
            "no relevant information",
            "nothing relevant",
            "not enough information",
        )
        if "none" in lowered_output and any(
            marker in lowered_output for marker in negative_markers
        ):
            return self._CANONICAL_NONE

        return stripped_output

    def _assemble_prompt_messages(
        self,
        *,
        system_messages: list[str],
        finalized_history: list[MessageRecord],
        current_user_message: str,
        requested_max_tokens: int,
    ) -> tuple[PromptMessage, ...]:
        fixed_messages = [
            PromptMessage(role="system", content=message.strip())
            for message in system_messages
            if message.strip()
        ]
        history_messages = [
            PromptMessage(role=message.role, content=message.content)
            for message in finalized_history
        ]
        current_user_prompt = PromptMessage(role="user", content=current_user_message)

        candidate_history = history_messages[:]
        while True:
            prompt_messages = tuple(
                [*fixed_messages, *candidate_history, current_user_prompt]
            )
            prompt_token_count = self._count_prompt_tokens(prompt_messages)
            if self._prompt_fits(
                prompt_token_count,
                requested_max_tokens=requested_max_tokens,
            ):
                return prompt_messages

            if not candidate_history:
                raise SessionCapacityError(
                    "The assembled prompt is too large for the configured context window."
                )
            candidate_history = candidate_history[1:]

    def _generate_pass_output(
        self,
        pass_name: str,
        prompt_messages: tuple[PromptMessage, ...],
        *,
        requested_max_tokens: int,
    ) -> tuple[str, int]:
        prompt_tokens = self._build_prompt_tokens(prompt_messages)
        max_reply_tokens = self._available_reply_tokens(
            len(prompt_tokens),
            requested_max_tokens=requested_max_tokens,
        )
        reply = self._generate_completion(
            prompt_tokens=prompt_tokens,
            max_reply_tokens=max_reply_tokens,
        )
        completion_tokens = len(self._tokenize_text(reply)) if reply else 0
        return reply, len(prompt_tokens) + completion_tokens

    def _count_prompt_tokens(
        self,
        prompt_messages: tuple[PromptMessage, ...],
    ) -> int:
        return len(self._build_prompt_tokens(prompt_messages))

    def _prompt_fits(
        self,
        prompt_token_count: int,
        *,
        requested_max_tokens: int,
    ) -> bool:
        return prompt_token_count + min(requested_max_tokens, self._settings.max_tokens) + 1 < self._settings.model_n_ctx

    def _build_prompt_tokens(
        self,
        prompt_messages: tuple[PromptMessage, ...],
    ) -> list[int]:
        if not prompt_messages:
            raise RuntimeError("Prompt assembly produced no messages.")

        tokens = list(self._bos_tokens)
        for index, message in enumerate(prompt_messages):
            tokens.extend(self._prefix_tokens_for_role(message.role))
            tokens.extend(self._tokenize_text(message.content.strip()))
            tokens.extend(self._eot_tokens)
            if index == len(prompt_messages) - 1:
                tokens.extend(self._assistant_prefix_tokens)
        return tokens

    def _available_reply_tokens(
        self,
        prompt_token_count: int,
        *,
        requested_max_tokens: int,
    ) -> int:
        available_reply_tokens = self._settings.model_n_ctx - prompt_token_count - 1
        minimum_reply_tokens = min(32, requested_max_tokens, self._settings.max_tokens)
        if available_reply_tokens <= 0 or available_reply_tokens < minimum_reply_tokens:
            raise SessionCapacityError(
                "The assembled prompt is out of context space. Reset or shorten the conversation to continue."
            )
        return min(requested_max_tokens, self._settings.max_tokens, available_reply_tokens)

    def _generate_completion(
        self,
        *,
        prompt_tokens: list[int],
        max_reply_tokens: int,
    ) -> str:
        assert self._llm is not None
        assert self._eot_token_id is not None

        completion_tokens: list[int] = []
        self._llm.reset()
        generator = self._llm.generate(
            prompt_tokens,
            temp=self._settings.temperature,
            top_p=self._settings.top_p,
            reset=True,
        )

        try:
            for token in generator:
                if token == self._eot_token_id:
                    break
                completion_tokens.append(token)
                if len(completion_tokens) >= max_reply_tokens:
                    break
        finally:
            generator.close()

        return self._llm.detokenize(completion_tokens).decode(
            "utf-8",
            errors="ignore",
        ).strip()

    def _persist_successful_turn(
        self,
        *,
        conversation_id: str | None,
        user_message: str,
        final_reply: str,
        memory_result: MemoryRetrievalResult,
        draft_pass: PassResult,
        analysis_pass: PassResult,
        final_pass: PassResult,
    ) -> ConversationThreads:
        if self._conversation_repository is None or self._memory_store is None:
            raise RuntimeError("Conversation persistence is not configured.")

        user_created_at = self._utc_now()
        final_reply_created_at = self._utc_now()
        user_sentence_texts = [fragment.text for fragment in split_sentences(user_message)]
        assistant_sentence_texts = [
            fragment.text for fragment in split_sentences(final_reply)
        ]
        staged_point_ids: list[str] = []

        try:
            with self._conversation_repository.transaction() as connection:
                conversation = self._conversation_repository.threads.get_or_create_conversation(
                    conversation_id=conversation_id,
                    connection=connection,
                )
                if conversation.finalized.title is None:
                    self._conversation_repository.threads.set_conversation_title(
                        conversation.conversation_id,
                        self._derive_conversation_title(user_message),
                        connection=connection,
                    )
                trace_thread_id = conversation.trace.id
                finalized_thread_id = conversation.finalized.id

                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="user",
                    kind=self._TRACE_USER_KIND,
                    content=user_message,
                    connection=connection,
                    created_at=user_created_at,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="system",
                    kind=self._TRACE_RETRIEVED_MEMORY_KIND,
                    content=memory_result.memory_block or "Relevant memory:\nNone.",
                    connection=connection,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="system",
                    kind=self._TRACE_DRAFT_INPUT_KIND,
                    content=draft_pass.prompt_trace,
                    connection=connection,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="assistant",
                    kind=self._TRACE_DRAFT_OUTPUT_KIND,
                    content=draft_pass.output_text,
                    connection=connection,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="system",
                    kind=self._TRACE_ANALYSIS_INPUT_KIND,
                    content=analysis_pass.prompt_trace,
                    connection=connection,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="assistant",
                    kind=self._TRACE_ANALYSIS_OUTPUT_KIND,
                    content=analysis_pass.output_text,
                    connection=connection,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="system",
                    kind=self._TRACE_FINAL_INPUT_KIND,
                    content=final_pass.prompt_trace,
                    connection=connection,
                )
                self._conversation_repository.messages.add_trace_message(
                    trace_thread_id,
                    role="assistant",
                    kind=self._TRACE_FINAL_OUTPUT_KIND,
                    content=final_reply,
                    connection=connection,
                    created_at=final_reply_created_at,
                )

                finalized_user_message = (
                    self._conversation_repository.messages.add_finalized_user_message(
                        finalized_thread_id,
                        user_message,
                        connection=connection,
                        created_at=user_created_at,
                    )
                )
                user_sentence_rows = (
                    self._conversation_repository.sentences.add_finalized_message_sentences(
                        message_id=finalized_user_message.id,
                        thread_id=finalized_thread_id,
                        role="user",
                        sentences=user_sentence_texts,
                        connection=connection,
                        created_at=user_created_at,
                    )
                )

                indexed_user_sentences = [
                    IndexedUserSentence(
                        qdrant_point_id=str(uuid4()),
                        sentence_id=sentence.id,
                        thread_id=sentence.thread_id,
                        message_id=sentence.message_id,
                        sentence_index=sentence.sentence_index,
                        text=sentence.text,
                        created_at=sentence.created_at,
                    )
                    for sentence in user_sentence_rows
                ]
                staged_point_ids = [
                    sentence.qdrant_point_id for sentence in indexed_user_sentences
                ]
                if indexed_user_sentences:
                    self._memory_store.upsert_user_sentences(indexed_user_sentences)
                    self._conversation_repository.sentences.set_qdrant_point_ids(
                        {
                            sentence.sentence_id: sentence.qdrant_point_id
                            for sentence in indexed_user_sentences
                        },
                        connection=connection,
                    )

                finalized_assistant_message = (
                    self._conversation_repository.messages.add_finalized_assistant_response(
                        finalized_thread_id,
                        final_reply,
                        connection=connection,
                        created_at=final_reply_created_at,
                    )
                )
                self._conversation_repository.sentences.add_finalized_message_sentences(
                    message_id=finalized_assistant_message.id,
                    thread_id=finalized_thread_id,
                    role="assistant",
                    sentences=assistant_sentence_texts,
                    connection=connection,
                    created_at=final_reply_created_at,
                )
                return conversation
        except Exception:
            if staged_point_ids:
                try:
                    self._memory_store.delete_points(staged_point_ids)
                except Exception:
                    pass
            raise

    def _get_turn_count(self, conversation: ConversationThreads | None) -> int:
        if conversation is None or self._conversation_repository is None:
            return 0
        messages = self._conversation_repository.messages.list_messages(
            conversation.finalized.id
        )
        return sum(1 for message in messages if message.role == "assistant")

    def _render_prompt_trace(
        self,
        pass_name: str,
        prompt_messages: tuple[PromptMessage, ...],
    ) -> str:
        lines = [f"{pass_name} prompt:"]
        for message in prompt_messages:
            lines.append(f"{message.role}:\n{message.content}")
        return "\n\n".join(lines)

    def _prefix_tokens_for_role(self, role: str) -> tuple[int, ...]:
        if role == "system":
            return self._system_prefix_tokens
        if role == "user":
            return self._user_prefix_tokens
        if role == "assistant":
            return self._assistant_prefix_tokens
        raise ValueError(f"Unsupported prompt role: {role}")

    def _tokenize_special(self, value: str) -> list[int]:
        assert self._llm is not None
        return self._llm.tokenize(
            value.encode("utf-8"),
            add_bos=False,
            special=True,
        )

    def _tokenize_text(self, value: str) -> list[int]:
        assert self._llm is not None
        return self._llm.tokenize(
            value.encode("utf-8"),
            add_bos=False,
            special=False,
        )

    def _utc_now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _derive_conversation_title(self, user_message: str) -> str:
        normalized = " ".join(user_message.split())
        if len(normalized) <= 72:
            return normalized
        return f"{normalized[:69].rstrip()}..."
