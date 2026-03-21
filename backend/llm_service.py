from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from .config import Settings
from .model_catalog import ModelSpec
from .session_manager import SessionStatus, SingleSessionManager


class SessionCapacityError(RuntimeError):
    """Raised when the single live conversation runs out of context space."""


@dataclass(frozen=True)
class ChatTurnResult:
    reply: str
    model: str
    turn_count: int
    used_tokens: int
    context_window: int


class LlamaService:
    _LLAMA3_BOS = "<|begin_of_text|>"
    _HEADER_TEMPLATE = "<|start_header_id|>{role}<|end_header_id|>\n\n"
    _EOT = "<|eot_id|>"

    def __init__(self, settings: Settings, model_spec: ModelSpec) -> None:
        self._settings = settings
        self._model_spec = model_spec
        self._session_manager = SingleSessionManager(
            system_prompt=settings.system_prompt
        )
        self._load_lock = Lock()
        self._llm: Llama | None = None
        self._model_path: str | None = None
        self._system_prefix_tokens: tuple[int, ...] = ()
        self._user_prefix_tokens: tuple[int, ...] = ()
        self._assistant_prefix_tokens: tuple[int, ...] = ()
        self._eot_tokens: tuple[int, ...] = ()
        self._eot_token_id: int | None = None

    @property
    def model_name(self) -> str:
        if self._model_path:
            return Path(self._model_path).name
        return self._settings.model_filename

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def session_status(self) -> SessionStatus:
        return self._session_manager.status(self._settings.model_n_ctx)

    def prepare_session(self) -> SessionStatus:
        with self._session_manager.lock:
            self.load()
            self._ensure_session_initialized()
            return self.session_status()

    def reset_session(self) -> SessionStatus:
        with self._session_manager.lock:
            if self._llm is None:
                self._session_manager.clear()
                return self.session_status()

            self._ensure_session_initialized()
            self._session_manager.reset()

            active_state = self._session_manager.active_state
            if active_state is not None:
                self._llm.load_state(active_state)

            return self.session_status()

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

    def generate_reply(self, user_message: str) -> ChatTurnResult:
        prompt = user_message.strip()
        if not prompt:
            raise ValueError("A message is required.")

        with self._session_manager.lock:
            self.load()
            self._ensure_session_initialized()
            assert self._llm is not None

            active_state = self._session_manager.active_state
            assert active_state is not None

            self._llm.load_state(active_state)

            turn_tokens = self._build_user_turn_tokens(prompt)
            max_reply_tokens = self._available_reply_tokens(len(turn_tokens))
            reply = self._generate_assistant_reply(
                turn_tokens=turn_tokens,
                max_reply_tokens=max_reply_tokens,
            )

            current_state = self._llm.save_state()
            self._session_manager.store_turn(current_state)

            return ChatTurnResult(
                reply=reply,
                model=self.model_name,
                turn_count=self._session_manager.turn_count,
                used_tokens=current_state.n_tokens,
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
            repo_id=self._settings.model_repo_id,
            filename=self._settings.model_filename,
            local_dir=str(cache_dir),
        )

    def _ensure_session_initialized(self) -> None:
        if self._session_manager.initialized:
            return

        assert self._llm is not None

        system_tokens = self._build_system_prompt_tokens(self._settings.system_prompt)
        if len(system_tokens) >= self._settings.model_n_ctx:
            raise RuntimeError(
                "The system prompt is too large for the configured context window."
            )

        self._llm.reset()
        self._llm.eval(system_tokens)
        self._session_manager.initialize(self._llm.save_state())

    def _prepare_template_tokens(self) -> None:
        self._system_prefix_tokens = tuple(
            self._tokenize_special(
                f"{self._LLAMA3_BOS}{self._HEADER_TEMPLATE.format(role='system')}"
            )
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

    def _build_system_prompt_tokens(self, system_prompt: str) -> list[int]:
        return [
            *self._system_prefix_tokens,
            *self._tokenize_text(system_prompt.strip()),
            *self._eot_tokens,
        ]

    def _build_user_turn_tokens(self, user_message: str) -> list[int]:
        return [
            *self._user_prefix_tokens,
            *self._tokenize_text(user_message),
            *self._eot_tokens,
            *self._assistant_prefix_tokens,
        ]

    def _available_reply_tokens(self, turn_token_count: int) -> int:
        assert self._llm is not None

        reserved_tokens = self._llm.n_tokens + turn_token_count + 1
        available_reply_tokens = self._settings.model_n_ctx - reserved_tokens

        if available_reply_tokens <= 0 or available_reply_tokens < min(
            32, self._settings.max_tokens
        ):
            raise SessionCapacityError(
                "The active conversation is out of context space. Reset the conversation to continue."
            )

        return min(self._settings.max_tokens, available_reply_tokens)

    def _generate_assistant_reply(
        self,
        turn_tokens: list[int],
        max_reply_tokens: int,
    ) -> str:
        assert self._llm is not None
        assert self._eot_token_id is not None

        completion_tokens: list[int] = []
        generator = self._llm.generate(
            turn_tokens,
            temp=self._settings.temperature,
            top_p=self._settings.top_p,
            reset=False,
        )

        reply_closed = False
        try:
            for token in generator:
                if token == self._eot_token_id:
                    reply_closed = True
                    break

                completion_tokens.append(token)
                if len(completion_tokens) >= max_reply_tokens:
                    break
        finally:
            generator.close()

        if not reply_closed:
            self._append_assistant_eot()

        return self._llm.detokenize(completion_tokens).decode(
            "utf-8", errors="ignore"
        ).strip()

    def _append_assistant_eot(self) -> None:
        assert self._llm is not None
        self._llm.eval(list(self._eot_tokens))

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
