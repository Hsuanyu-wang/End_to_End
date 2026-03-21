"""
LightRAGWrapper 官方端到端路徑（only_need_context=False）單元測試
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from src.rag.wrappers.lightrag_wrapper import LightRAGWrapper, LightRAGWrapper_Original


class TestLightRAGWrapperNative(unittest.TestCase):
    """驗證 _native_answer_query_param、query()、_execute_original_mode 行為"""

    def test_native_query_param_only_need_context_false(self):
        rag = MagicMock()
        w = LightRAGWrapper(name="t", rag_instance=rag, mode="local", use_context=False)
        p = w._native_answer_query_param()
        self.assertEqual(p.mode, "local")
        self.assertFalse(p.only_need_context)

    def test_query_raises_when_use_context_true(self):
        rag = MagicMock()
        w = LightRAGWrapper(name="t", rag_instance=rag, mode="hybrid", use_context=True)
        with self.assertRaises(NotImplementedError):
            w.query("q")

    def test_query_uses_native_param_when_use_context_false(self):
        rag = MagicMock()
        rag.query.return_value = "ans"
        w = LightRAGWrapper(name="t", rag_instance=rag, mode="mix", use_context=False)
        out = w.query("hello")
        self.assertEqual(out, "ans")
        rag.query.assert_called_once()
        args, kwargs = rag.query.call_args
        self.assertEqual(args[0], "hello")
        param = kwargs.get("param")
        self.assertEqual(param.mode, "mix")
        self.assertFalse(param.only_need_context)

    def test_execute_original_mode_passes_native_param_to_aquery(self):
        async def run():
            rag = MagicMock()
            rag.aquery = AsyncMock(return_value="async_ans")
            w = LightRAGWrapper_Original(name="t", rag_instance=rag, mode="global")
            out = await w._execute_original_mode("q")
            self.assertEqual(out, "async_ans")
            rag.aquery.assert_called_once()
            call_args, kwargs = rag.aquery.call_args
            self.assertEqual(call_args[0], "q")
            self.assertFalse(kwargs["param"].only_need_context)
            self.assertEqual(kwargs["param"].mode, "global")

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
