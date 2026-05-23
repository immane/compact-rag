from __future__ import annotations

from compact_rag.admin.client import AdminAPIClient


class TestAdminAPIClientProxyBehavior:
    def test_local_base_url_disables_env_proxy(self):
        client = AdminAPIClient("http://127.0.0.1:8000")
        assert client.session.trust_env is False

    def test_localhost_domain_disables_env_proxy(self):
        client = AdminAPIClient("http://api.localhost:8000")
        assert client.session.trust_env is False

    def test_remote_base_url_keeps_env_proxy(self):
        client = AdminAPIClient("https://api.example.com")
        assert client.session.trust_env is True


class TestAdminAPIClientStorageLinks:
    def test_storage_download_url_uses_download_mode(self):
        client = AdminAPIClient("http://127.0.0.1:8000")

        assert (
            client.get_file_url("docs/default/2026/05/24/my file.pdf")
            == "http://127.0.0.1:8000/v1/files/docs/default/2026/05/24/my%20file.pdf?download=true"
        )
