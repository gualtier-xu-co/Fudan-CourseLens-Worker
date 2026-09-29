"""夜10-C T6: dispatch/failure/retry 合成矩阵扩充 — 错误码闭集净化矩阵、
source-session 收口排列、进度发布器闭集状态 coercion。

零真实端点、零媒体、零模型依赖；全部走 mock 与纯函数面。
"""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

with patch.dict("sys.modules", {"sherpa_onnx": Mock()}):
    from courselens_worker.asr import ASRError
    from courselens_worker.platform_session import PlatformSessionError
from courselens_worker.protocol import (
    JOB_SCHEMA,
    PROTOCOL_VERSION,
)
from courselens_worker.runner import (
    SignedProgressPublisher,
    process_job,
    safe_worker_error_detail,
)


class SafeWorkerErrorDetailMatrixTests(unittest.TestCase):
    """错误码闭集净化矩阵：未知码塌缩、registered stage 才允许后缀、
    未知异常返回空串（run() 侧再塌缩成 worker_failed）。"""

    def test_platform_challenge_required_stays_independent_without_suffix(self):
        error = PlatformSessionError(
            "platform_challenge_required", connection_stage="course_context"
        )
        self.assertEqual(safe_worker_error_detail(error), "platform_challenge_required")

    def test_connection_failure_gets_registered_stage_suffix(self):
        error = PlatformSessionError(
            "platform_connection_failed", connection_stage="webvpn_context"
        )
        self.assertEqual(
            safe_worker_error_detail(error),
            "platform_connection_failed_webvpn_context",
        )

    def test_unregistered_stage_never_leaks_into_the_code(self):
        error = PlatformSessionError(
            "platform_connection_failed", connection_stage="totally_bogus_stage"
        )
        self.assertEqual(safe_worker_error_detail(error), "platform_connection_failed")

    def test_unknown_platform_code_collapses_to_session_failed(self):
        error = PlatformSessionError("platform_some_future_code")
        self.assertEqual(safe_worker_error_detail(error), "platform_session_failed")

    def test_asr_error_matrix_maps_known_and_collapses_unknown(self):
        known = ASRError("authorized media request returned HTTP 403")
        self.assertEqual(safe_worker_error_detail(known), "media_http_403")
        unknown = ASRError("some brand new asr failure text")
        self.assertEqual(safe_worker_error_detail(unknown), "asr_error")

    def test_generic_exception_returns_empty_detail(self):
        self.assertEqual(safe_worker_error_detail(RuntimeError("boom")), "")


class SourceSessionCloseMatrixTests(unittest.TestCase):
    """process_job 的 source-session 收口排列：无论检查点/进度回调怎么炸，
    收口回调恰一次；无会话载荷绝不误收口。"""

    def _job(self, close=None):
        payload = {}
        if close is not None:
            payload["_close_source_session"] = close
        return {
            "schema": JOB_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "task_id": "0123456789abcdef0123456789abcdef",
            "job_kind": "echo",
            "input_hash": "0" * 64,
            "pipeline": {"version": "actions-echo-v2"},
            "payload": payload,
        }

    def test_close_runs_once_when_checkpoint_writer_explodes(self):
        close = Mock()
        with patch(
            "courselens_worker.runner._process_materialized_job",
            side_effect=lambda job, **kwargs: kwargs["checkpoint_writer"](
                {"stage": "half", "completed_chunks": 1}
            ),
        ):
            with self.assertRaises(Exception):
                process_job(self._job(close), checkpoint_writer=RuntimeError("boom"))
        close.assert_called_once_with()

    def test_close_runs_once_when_progress_callback_explodes(self):
        close = Mock()
        with patch(
            "courselens_worker.runner._process_materialized_job",
            side_effect=lambda job, **kwargs: kwargs["progress_callback"](
                "asr", 1, 2
            ),
        ):
            with self.assertRaises(Exception):
                process_job(
                    self._job(close), progress_callback=RuntimeError("boom")
                )
        close.assert_called_once_with()

    def test_close_is_skipped_without_session_payload(self):
        close = Mock()
        with patch(
            "courselens_worker.runner._process_materialized_job",
            return_value={"status": "completed", "outputs": {}, "metrics": {}},
        ):
            process_job(self._job(None))
        close.assert_not_called()

    def test_success_result_passes_through_unchanged(self):
        with patch(
            "courselens_worker.runner._process_materialized_job",
            return_value={"status": "completed", "outputs": {"echo": {"ok": True}}},
        ) as inner:
            result = process_job(self._job(None))
        self.assertEqual(result["status"], "completed")
        inner.assert_called_once()


class SignedProgressPublisherClosedSetTests(unittest.TestCase):
    def test_unknown_status_coerces_to_running(self):
        messages = []
        publisher = SignedProgressPublisher(messages.append, heartbeat_seconds=60)
        publisher.update("asr", status="suspicious-value", force=True)
        publisher.close()
        self.assertEqual(messages[-1]["status"], "running", "闭集外状态塌缩为 running")

    def test_all_closed_set_statuses_pass_through(self):
        for status in ("running", "waiting", "failed", "completed"):
            messages = []
            publisher = SignedProgressPublisher(messages.append, heartbeat_seconds=60)
            publisher.update("asr", status=status, force=True)
            publisher.close()
            self.assertEqual(messages[-1]["status"], status)


if __name__ == "__main__":
    unittest.main()
