import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from assistant.database import Repository
from assistant.engine import AssistantEngine
from assistant.engine import speech_friendly_text
from assistant.semantic_search import SemanticSearch


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
        result = engine.handle("Добавь задачу на завтра в 10:00: сдать доклад")
        self.assertIn("Создано", result)
        row = self.repo.all()[0]
        self.assertEqual(row["due_at"], "2026-10-21T10:00")

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
        self.assertEqual(engine._parse_date("2026-10-21"), "2026-10-21T00:00")

    def test_create_accepts_iso_datetime_from_llm(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Проверить расписание", "date_text": "2026-10-21T13:57", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на завтра в 10:00: проверить расписание")
        self.assertIn("Создано", result)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-21T10:00")

    def test_create_relative_date_overrides_stale_llm_date(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "задача на завтра - сходить в парк", "date_text": "2026-10-19T00:00", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Создать задачу на завтра в 10:00 сходить в парк")
        row = self.repo.all()[0]
        self.assertIn("Создано", result)
        self.assertEqual(row["title"], "Сходить в парк")
        self.assertEqual(row["due_at"], "2026-10-21T10:00")

    def test_create_title_removes_command_and_date_phrase(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "задачу на завтра проверить расписание", "date_text": "завтра", "period": "week", "target_id": None, "clarification": ""}])
        engine.handle("Добавь задачу на завтра в 10:00: проверить расписание")
        self.assertEqual(self.repo.all()[0]["title"], "Проверить расписание")

    def test_bare_tasks_for_tomorrow_is_list(self):
        self.repo.create("task", "Сходить в парк", "2026-10-21T09:00")
        engine = self.engine([{"action": "search", "kind": "any", "title": "задачи на завтра", "date_text": "", "period": "all", "target_id": None, "clarification": ""}])
        result = engine.handle("задачи на завтра")
        self.assertIn("Сходить в парк", result)

    def test_create_uses_date_from_original_text_when_llm_omits_it(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Проверить расписание", "date_text": "", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на завтра в 10:00: проверить расписание")
        self.assertIn("Создано", result)
        row = self.repo.all()[0]
        self.assertEqual(row["due_at"], "2026-10-21T10:00")

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
        first = engine.handle("Добавь задачу на завтра в 10:00: проверить расписание")
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


    def test_vague_daypart_asks_for_exact_time(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Погулять на речке", "date_text": "сегодня", "period": "", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу сегодня в обед: погулять на речке")
        self.assertIn("точное время", result)
        self.assertEqual(self.repo.all(), [])
        result = engine.handle("13:00")
        self.assertIn("Создано", result)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-20T13:00")

    def test_create_on_15_hours_today_keeps_time_and_cleans_title(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "На 15 часов, сходите на работу", "date_text": "", "period": "week", "target_id": None, "clarification": ""}])
        result = engine.handle("Добавь задачу на 15 часов, на сегодня сходите на работу")
        row = self.repo.all()[0]
        self.assertIn("Создано", result)
        self.assertEqual(row["title"], "Сходите на работу")
        self.assertEqual(row["due_at"], "2026-10-20T15:00")

    def test_parse_dayparts(self):
        engine = self.engine([])
        self.assertEqual(engine._parse_date("сегодня в обед"), "2026-10-20T00:00")
        self.assertEqual(engine._parse_date("завтра вечером"), "2026-10-21T00:00")
        self.assertEqual(engine._parse_date("сегодня в 15:00"), "2026-10-20T15:00")

    def test_empty_title_rejected(self):
        with self.assertRaises(ValueError):
            self.repo.create("task", "   ")

    def test_llm_clarify_classification_still_starts_meeting_dialog(self):
        engine = self.engine([{"action": "clarify", "kind": "meeting", "title": "Встреча с Иваном", "date_text": "завтра", "period": "week", "target_id": None, "clarification": "Уточните время"}])
        first = engine.handle("Добавь встречу с Иваном завтра")
        self.assertIn("точное время", first)
        self.assertEqual(self.repo.all(), [])
        second = engine.handle("в 15:00")
        self.assertIn("Создано", second)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-21T15:00")

    def test_meeting_missing_time_starts_active_clarification(self):
        engine = self.engine([{"action": "create", "kind": "meeting", "title": "Встреча с Иваном", "date_text": "завтра", "period": "week", "target_id": None, "clarification": ""}])
        first = engine.handle("Добавь встречу с Иваном завтра")
        self.assertIn("точное время", first)
        self.assertEqual(self.repo.all(), [])
        second = engine.handle("в 15:00")
        self.assertIn("Создано", second)
        row = self.repo.all()[0]
        self.assertEqual(row["title"], "Встреча с Иваном")
        self.assertEqual(row["due_at"], "2026-10-21T15:00")

    def test_meeting_clarification_can_supply_date_and_time(self):
        engine = self.engine([{"action": "create", "kind": "meeting", "title": "Дизайнер", "date_text": "", "period": "week", "target_id": None, "clarification": ""}])
        first = engine.handle("Добавь встречу с дизайнером")
        self.assertIn("На какую дату", first)
        second = engine.handle("в пятницу в 11:30")
        self.assertIn("Создано", second)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-23T11:30")

    def test_semantic_search_cosine_ranking_with_injected_encoder(self):
        # Query is close to the first row; the second row is orthogonal.
        vectors = {
            "когда обсуждали дизайн": [1.0, 0.0],
            "meeting. Обсудить макет": [0.9, 0.1],
            "task. Купить молоко": [0.0, 1.0],
        }
        search = SemanticSearch(encoder=lambda texts: [vectors[text] for text in texts])
        rows = [
            {"id": 1, "kind": "meeting", "title": "Обсудить макет"},
            {"id": 2, "kind": "task", "title": "Купить молоко"},
        ]
        result = search.rank("когда обсуждали дизайн", rows)
        self.assertEqual([row["id"] for row in result], [1])

    def test_semantic_search_results_are_added_to_lexical_search(self):
        row = self.repo.create("meeting", "Обсудить макет с Иваном", "2026-10-21T15:00")
        class FakeSemantic:
            def rank(self, query, rows, limit=10):
                self.assertion = query
                return [r for r in rows if r["id"] == row["id"]]
        semantic = FakeSemantic()
        engine = AssistantEngine(FakeLLM([{"action": "search", "kind": "any", "title": "дизайн", "date_text": "", "period": "all", "target_id": None, "clarification": ""}]), self.repo, clock=lambda: self.fixed_now, semantic_search=semantic)
        result = engine.handle("Когда мы обсуждали дизайн с Иваном?")
        self.assertIn("Обсудить макет с Иваном", result)
        self.assertEqual(semantic.assertion, "Когда мы обсуждали дизайн с Иваном?")

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


    def test_task_without_time_asks_and_does_not_create_until_answer(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Сходить в парк", "date_text": "завтра", "period": "", "target_id": None, "clarification": ""}])
        first = engine.handle("Добавь задачу на завтра сходить в парк")
        self.assertIn("точное время", first)
        self.assertEqual(self.repo.all(), [])
        second = engine.handle("в 15:30")
        self.assertIn("Создано", second)
        self.assertEqual(self.repo.all()[0]["title"], "Сходить в парк")
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-21T15:30")

    def test_task_without_date_asks_for_date_and_then_time(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Купить хлеб", "date_text": "", "period": "", "target_id": None, "clarification": ""}])
        first = engine.handle("Добавь задачу купить хлеб в 16:00")
        self.assertIn("На какую дату", first)
        self.assertEqual(self.repo.all(), [])
        second = engine.handle("завтра")
        self.assertIn("Создано", second)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-21T16:00")

    def test_vague_daypart_is_not_silently_converted_to_time(self):
        engine = self.engine([])
        self.assertEqual(engine._extract_explicit_time("завтра вечером"), None)
        self.assertEqual(engine._extract_date_phrase("завтра вечером"), "завтра")

    def test_list_without_period_asks_instead_of_defaulting_to_week(self):
        engine = self.engine([{"action": "list", "kind": "any", "title": "", "date_text": "", "period": "", "target_id": None, "clarification": ""}])
        result = engine.handle("Покажи мои записи")
        self.assertIn("За какой период", result)


    def test_task_without_date_or_time_collects_both_before_saving(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Позвонить Ивану", "date_text": "", "period": "", "target_id": None, "clarification": ""}])
        first = engine.handle("Добавь задачу позвонить Ивану")
        self.assertIn("На какую дату", first)
        self.assertEqual(self.repo.all(), [])
        second = engine.handle("послезавтра")
        self.assertIn("точное время", second)
        self.assertEqual(self.repo.all(), [])
        third = engine.handle("в 14:20")
        self.assertIn("Создано", third)
        self.assertEqual(self.repo.all()[0]["due_at"], "2026-10-22T14:20")

    def test_clarification_can_be_cancelled_without_creating(self):
        engine = self.engine([{"action": "create", "kind": "task", "title": "Сходить в магазин", "date_text": "завтра", "period": "", "target_id": None, "clarification": ""}])
        engine.handle("Добавь задачу на завтра сходить в магазин")
        result = engine.handle("отмена")
        self.assertIn("отменено", result.casefold())
        self.assertEqual(self.repo.all(), [])

if __name__ == "__main__":
    unittest.main(verbosity=2)
