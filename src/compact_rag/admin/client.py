"""Admin API client using requests for Streamlit compatibility (sync)."""

from __future__ import annotations

import json
from urllib.parse import quote, urlparse

import requests


DEFAULT_REQUEST_TIMEOUT = (10, 120)
UPLOAD_REQUEST_TIMEOUT = (10, 300)
STREAM_REQUEST_TIMEOUT = (10, 300)


def _is_local_base_url(base_url: str) -> bool:
    host = urlparse(base_url).hostname
    if not host:
        return False
    return host in {"127.0.0.1", "localhost", "::1"} or host.endswith(".localhost")


class AdminAPIClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        if _is_local_base_url(self.base_url):
            # Avoid routing localhost calls via HTTP(S)_PROXY, which can cause false 502s.
            self.session.trust_env = False

    def _get(self, path: str, **kwargs) -> dict:
        r = self.session.get(
            f"{self.base_url}{path}", timeout=DEFAULT_REQUEST_TIMEOUT, **kwargs
        )
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, **kwargs) -> dict:
        headers = kwargs.pop("headers", {})
        headers.setdefault("Content-Type", "application/json")
        r = self.session.post(
            f"{self.base_url}{path}",
            timeout=DEFAULT_REQUEST_TIMEOUT,
            headers=headers,
            **kwargs,
        )
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, **kwargs) -> dict:
        headers = kwargs.pop("headers", {})
        headers.setdefault("Content-Type", "application/json")
        r = self.session.patch(
            f"{self.base_url}{path}",
            timeout=DEFAULT_REQUEST_TIMEOUT,
            headers=headers,
            **kwargs,
        )
        r.raise_for_status()
        return r.json()

    def _delete(self, path: str) -> dict:
        r = self.session.delete(
            f"{self.base_url}{path}", timeout=DEFAULT_REQUEST_TIMEOUT
        )
        r.raise_for_status()
        return r.json()

    # ── System ──────────────────────────────────────────────

    def health(self) -> dict:
        return self._get("/v1/health")

    def info(self) -> dict:
        return self._get("/v1/info")

    # ── Collections ─────────────────────────────────────────

    def list_collections(self, page: int = 1, page_size: int = 20) -> dict:
        return self._get(
            "/v1/collections", params={"page": page, "page_size": page_size}
        )

    def create_collection(
        self,
        name: str,
        description: str = "",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> dict:
        return self._post(
            "/v1/collections",
            json={
                "name": name,
                "description": description,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
            },
        )

    def delete_collection(self, name: str) -> dict:
        return self._delete(f"/v1/collections/{name}")

    # ── Documents ───────────────────────────────────────────

    def list_documents(
        self,
        collection: str | None = None,
        status: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        params: dict[str, str | int] = {"page": page, "page_size": page_size}
        if collection:
            params["collection"] = collection
        if status:
            params["status"] = status
        return self._get("/v1/documents", params=params)

    def upload_document(
        self, file_data: bytes, filename: str, collection: str = "default"
    ) -> dict:
        r = self.session.post(
            f"{self.base_url}/v1/documents/ingest",
            files={"file": (filename, file_data)},
            data={"collection": collection},
            timeout=UPLOAD_REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        return r.json()

    def get_document(self, doc_id: str) -> dict:
        return self._get(f"/v1/documents/{doc_id}")

    def delete_document(self, doc_id: str) -> dict:
        return self._delete(f"/v1/documents/{doc_id}")

    # ── Ingestion ───────────────────────────────────────────

    def list_ingestion_jobs(
        self,
        status: str | None = None,
        collection: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> dict:
        params: dict[str, str | int] = {"page": page, "page_size": page_size}
        if status:
            params["status"] = status
        if collection:
            params["collection"] = collection
        return self._get("/v1/ingestion-jobs", params=params)

    def get_ingestion_job(self, job_id: str) -> dict:
        return self._get(f"/v1/ingestion-jobs/{job_id}")

    # ── Conversations ───────────────────────────────────────

    def list_conversations(self, page: int = 1, page_size: int = 20) -> dict:
        return self._get(
            "/v1/conversations", params={"page": page, "page_size": page_size}
        )

    def get_conversation(self, conv_id: str) -> dict:
        return self._get(f"/v1/conversations/{conv_id}")

    def delete_conversation(self, conv_id: str) -> dict:
        return self._delete(f"/v1/conversations/{conv_id}")

    # ── Chat ────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        collection: str = "default",
        top_k: int = 10,
        temperature: float = 0.1,
        use_rerank: bool = True,
        use_hybrid: bool = True,
    ) -> dict:
        return self._post(
            "/v1/chat/completions",
            json={
                "messages": messages,
                "collection": collection,
                "retrieval": {
                    "top_k": top_k,
                    "rerank": use_rerank,
                    "hybrid_search": use_hybrid,
                },
                "temperature": temperature,
                "stream": False,
            },
        )

    def chat_stream(
        self,
        messages: list[dict],
        collection: str = "default",
        top_k: int = 10,
        temperature: float = 0.1,
        use_rerank: bool = True,
        use_hybrid: bool = True,
    ):
        """Yield content chunks from SSE streaming response."""
        with self.session.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "collection": collection,
                "retrieval": {
                    "top_k": top_k,
                    "rerank": use_rerank,
                    "hybrid_search": use_hybrid,
                },
                "temperature": temperature,
                "stream": True,
            },
            stream=True,
            timeout=STREAM_REQUEST_TIMEOUT,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line and line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta and delta["content"]:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    # ── API Keys ────────────────────────────────────────────

    def list_api_keys(self, page: int = 1, page_size: int = 20) -> dict:
        return self._get("/v1/api-keys", params={"page": page, "page_size": page_size})

    def create_api_key(self, name: str, permissions: list[str] | None = None) -> dict:
        return self._post(
            "/v1/api-keys",
            json={
                "name": name,
                "permissions": permissions or ["read"],
            },
        )

    def toggle_api_key(self, key_id: str, active: bool) -> dict:
        return self._patch(f"/v1/api-keys/{key_id}", json={"is_active": active})

    def delete_api_key(self, key_id: str) -> dict:
        return self._delete(f"/v1/api-keys/{key_id}")

    # ── Storage ─────────────────────────────────────────────

    def list_storage_files(self) -> dict:
        return self._get("/v1/files")

    def get_file_url(self, storage_key: str) -> str:
        encoded_key = quote(storage_key, safe="/")
        return f"{self.base_url}/v1/files/{encoded_key}?download=true"

    def delete_storage_file(self, storage_key: str) -> dict:
        encoded_key = quote(storage_key, safe="/")
        return self._delete(f"/v1/files/{encoded_key}")

    def clean_temp_files(self) -> dict:
        return self._post("/v1/files/clean-temp")
