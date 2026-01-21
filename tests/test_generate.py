import rag_system.generate as generate


def test_generate_prints_stream(monkeypatch, capsys):
    monkeypatch.setattr(generate, "LANGUAGE_MODEL", "fake-model")

    def fake_chat(model, messages, stream):
        assert stream is True
        yield {"message": {"content": "Hello"}}
        yield {"message": {"content": " world"}}

    monkeypatch.setattr(generate.ollama, "chat", fake_chat)

    generate.generate("question", context=[("ctx1", 0.1), ("ctx2", 0.2)])
    out = capsys.readouterr().out
    assert "Chatbot response:" in out
    assert "Hello world" in out
