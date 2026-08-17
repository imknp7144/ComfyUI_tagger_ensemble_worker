"""
tew_utils/file_lock.py

指示書17: models.json は node_setup.py / vram_manager.py 等、複数の書き込み元が
read-modify-write するため、並列実行時に更新が消える(lost update)可能性がある。

ComfyUI実行環境はWindows(StabilityMatrix同梱)を主対象とするため、fcntl/msvcrt等の
プラットフォーム依存APIに頼らず、「排他的にロックファイルを作成できるかどうか」
(`os.open` + `O_CREAT|O_EXCL`)だけを使った最小限のポータブルなファイルロックを実装する。
"""

from __future__ import annotations

import contextlib
import logging
import os
import time

logger = logging.getLogger("ComfyUI_Tagger_Ensemble_Worker")

_DEFAULT_TIMEOUT_SEC = 10.0
_POLL_INTERVAL_SEC = 0.05
_STALE_LOCK_SEC = 30.0  # 異常終了等で残留したロックファイルを無視するまでの秒数


@contextlib.contextmanager
def file_lock(target_path: str, timeout: float = _DEFAULT_TIMEOUT_SEC):
    """
    target_path + ".lock" の排他作成をロックとして使うコンテキストマネージャ。

    使い方:
        with file_lock(models_json_path):
            data = _load_all()
            data[...] = ...
            _save_all(data)
    """
    lock_path = target_path + ".lock"
    deadline = time.monotonic() + timeout
    acquired = False

    while time.monotonic() < deadline:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > _STALE_LOCK_SEC:
                    logger.warning(
                        "残留ロックファイルを検出しました(%.1f秒経過): %s。破棄して取得を試みます",
                        age, lock_path,
                    )
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            time.sleep(_POLL_INTERVAL_SEC)

    if not acquired:
        logger.warning(
            "ファイルロックの取得がタイムアウトしました(%.1f秒): %s。"
            "ロック無しで続行しますが、並列書き込みが競合する可能性があります",
            timeout, lock_path,
        )

    try:
        yield
    finally:
        if acquired:
            try:
                os.remove(lock_path)
            except OSError:
                logger.debug("ロックファイルの削除に失敗しました: %s", lock_path, exc_info=True)
