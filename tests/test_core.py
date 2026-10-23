import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant.database import Repository
from assistant.engine import AssistantEngine
from assistant.engine import speech_friendly_text


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)

    def parse(self, user_text, now_iso, records_summary=""):
        if not self.responses:
            raise AssertionError("Unexpected extra LLM parse call")
        return self.responses.pop(0)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Repository(Path(self.tmp.name) / "test.sqlite3")
        self.fixed_now = datetime(2026, 10, 20, 13, 0)

    def tearDown(self):
        self.repo.close()
        self.tmp.cleanup()

    def engine(self, responses):
        return AssistantEngine(FakeLLM(responses), self.repo, clock=lambda: self.fixed_now)

    def test_create_and_persist_across_repository_instances(self):
        row = self.repo.create("task", "Сдать доклад", "2026-10-21T09:00")
        self.assertEqual(row["title"], "Сдать доклад")
        second_repo = Repository(self.repo.db_path)
        self.assertEqual(second_repo.get(row["id"])["title"], "Сдать доклад")

    def test_create_request_parses_tomorrow(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Сдать доклад", "date_text": "завтра", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на завтра: сдать доклад")
        self.assertIn("Создано", result)
        row = self.repo.all()[0]
        self.assertEqual(row["due_at"], "2026-10-21T09:00")

    def test_list_and_search(self):
        self.repo.create("task", "Подготовить презентацию", "2026-10-21T10:00")
        list_engine = self.engine([{"action": "list", "kind": "any", "title": "", "date_text": "", "period": "tomorrow", "target_id": None, "clarification": ""}])
        self.assertIn("презентацию", list_engine.handle("Что у меня завтра?"))
        search_engine = self.engine([{"action": "search", "kind": "any", "title": "презентация", "date_text": "", "period": "all", "target_id": None, "clarification": ""}])
        self.assertIn("Подготовить презентацию", search_engine.handle("Найди презентацию"))

    def test_complete_and_delete(self):
        row = self.repo.create("task", "Позвонить Ивану")
        engine = self.engine([{"action": "complete", "kind": "task", "title": "Позвонить Ивану", "date_text": "", "period": "all", "target_id": row["id"], "clarification": ""}])
        self.assertIn("выполненным", engine.handle("Отметь выполненной задачу позвонить Ивану"))
        self.assertEqual(self.repo.get(row["id"])["status"], "done")
        delete_engine = self.engine([{"action": "delete", "kind": "task", "title": "Позвонить Ивану", "date_text": "", "period": "all", "target_id": row["id"], "clarification": ""}])
        self.assertIn("Удалено", delete_engine.handle("Удали задачу"))
        self.assertIsNone(self.repo.get(row["id"]))

    def test_relative_date_parsing(self):
        engine = self.engine([])
        self.assertEqual(engine._parse_date("послезавтра в 15:30"), "2026-10-22T15:30")
        self.assertEqual(engine._parse_date("через 2 часа"), "2026-10-20T15:00")
        self.assertEqual(engine._parse_date("25.10.2026 в 14:00"), "2026-10-25T14:00")

    def test_iso_datetime_returned_by_llm_is_accepted(self):
        engine = self.engine([])
        self.assertEqual(engine._parse_date("2026-10-21T13:57"), "2026-10-21T13:57")
        self.assertEqual(engine._parse_date("2026-10-21"), "2026-10-21T09:00")

    def test_create_accepts_iso_datetime_from_llm(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Проверить расписание", "date_text": "2026-10-21T13:57", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на завтра: проверить расписание")
        self.assertIn("Создано", result)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-21T09:00")

    def test_create_relative_date_overrides_stale_llm_date(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "задача на завтра - сходить в парк", "date_text": "2026-10-19T00:00", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Создать задачу на завтра сходить в парк")
        row = self.repo.all()[0]
        self.assertIn("Создано", result)
        self.assertEqual(row["title"], "Сходить в парк")
        self.assertEqual(row["due_at"], "2026-10-21T09:00")

    def test_create_title_removes_command_and_date_phrase(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "задачу на завтра проверить расписание", "date_text": "завтра", "period": "week", "target_id": None, "clarification": ""}])
        engine.handle("Добавь задачу на завтра: проверить расписание")
        self.assertEqual(self.repo.all()[0]["title"], "Проверить расписание")

    def test_bare_tasks_for_tomorrow_is_list(self):
        self.repo.create("task", "Сходить в парк", "2026-10-21T09:00")
        engine = self.engine([{"action": "search", "kind": "any", "title": "задачи на завтра", "date_text": "", "period": "all", "target_id": None, "clarification": ""}])
        result = engine.handle("задачи на завтра")
        self.assertIn("Сходить в парк", result)

    def test_create_uses_date_from_original_text_when_llm_omits_it(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Проверить расписание", "date_text": "", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на завтра: проверить расписание")
        self.assertIn("Создано", result)
        row = self.repo.all()[0]
        self.assertEqual(row["due_at"], "2026-10-21T09:00")

    def test_tomorrow_question_is_list_even_if_llm_says_search(self):
        self.repo.create("task", "Проверить расписание", "2026-10-21T09:00")
        engine = self.engine([{"action": "search", "kind": "any", "title": "Какая задача на завтра?", "date_text": "", "period": "all", "target_id": None, "clarification": ""}])
        result = engine.handle("Какая задача на завтра?")
        self.assertIn("Проверить расписание", result)
        self.assertNotIn("Не нашёл запись", result)

    def test_two_requests_in_sequence(self):
        engine = self.engine([
            {"action": "create", "kind": "task", "title": "Проверить расписание", "date_text": "завтра", "period": "week", "target_id": None, "clarification": ""},
            {"action": "search", "kind": "any", "title": "расписание", "date_text": "", "period": "all", "target_id": None, "clarification": ""},
        ])
        first = engine.handle("Добавь задачу на завтра: проверить расписание")
        second = engine.handle("Найди задачу про расписание")
        self.assertIn("Создано", first)
        self.assertIn("проверить расписание", second.casefold())

    def test_reschedule(self):
        row = self.repo.create("meeting", "Встреча с командой", "2026-10-21T15:00")
        engine = self.engine([{"action": "reschedule", "kind": "meeting", "title": "Встреча с командой", "date_text": "пятницу в 14:00", "period": "all", "target_id": row["id"], "clarification": ""}])
        result = engine.handle("Перенеси встречу на пятницу в 14:00")
        self.assertIn("Перенесено", result)
        self.assertEqual(self.repo.get(row["id"])["due_at"], "2026-10-23T14:00")

    def test_display_date_is_russian_and_human_readable(self):
        engine = self.engine([])
        self.assertEqual(engine._format_due("2026-10-23T00:00"), "23 октября 2026 года, 00:00")

    def test_spoken_text_does_not_read_database_hash_id(self):
        spoken = speech_friendly_text("Создано: #5 задача: «Сходить в парк», срок: 2026-10-23T09:00, статус: активна")
        self.assertNotIn("#", spoken)
        self.assertIn("задача", spoken)
        self.assertIn("Сходить в парк", spoken)


    def test_create_lunch_phrase_uses_13_and_removes_daypart_from_title(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "В обед, садить, погулять на речку", "date_text": "", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу сегодня в обед, садить, погулять на речку")
        row = self.repo.all()[0]
        self.assertIn("Создано", result)
        self.assertEqual(row["title"], "Садить, погулять на речку")
        self.assertEqual(row["due_at"], "2026-10-20T13:00")

    def test_create_on_15_hours_today_keeps_time_and_cleans_title(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "На 15 часов, сходите на работу", "date_text": "", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на 15 часов, на сегодня сходите на работу")
        row = self.repo.all()[0]
        self.assertIn("Создано", result)
        self.assertEqual(row["title"], "Сходите на работу")
        self.assertEqual(row["due_at"], "2026-10-20T15:00")

    def test_parse_dayparts(self):
        engine = self.engine([])
        self.assertEqual(engine._parse_date("сегодня в обед"), "2026-10-20T13:00")
        self.assertEqual(engine._parse_date("завтра вечером"), "2026-10-21T18:00")
        self.assertEqual(engine._parse_date("сегодня в 15:00"), "2026-10-20T15:00")

    def test_empty_title_rejected(self):
        with self.assertRaises(ValueError):
            self.repo.create("task", "   ")

    def test_sequential_cross_thread_db_access(self):
        import threading
        errors = []
        def worker(i):
            try:
                self.repo.create("task", f"task-{i}")
                self.repo.all(10)
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(12)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(errors, [])
        self.assertEqual(len(self.repo.all(100)), 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
