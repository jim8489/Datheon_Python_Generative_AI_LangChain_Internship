# Applying this Task 02 / Task 04 fix pack

This archive is an **overlay pack** for the existing repository, not a standalone copy of every unchanged source file.

1. Back up your local repository or make a Git commit before applying it.
2. Extract this ZIP into the repository root:

   ```text
   Datheon_Python_Generative_AI_LangChain_Internship/
   ```

3. Allow the extracted `Task_02/`, `Task_04/`, and root `README.md` files to overwrite the matching files.
4. Keep your local `.env` files, `Task_02/chroma_db/`, `Task_04/data/`, SQLite database, and browser signing key. They are not included in this archive.
5. From `Task_02`, run:

   ```bash
   PYTHONPATH=. python -m pytest -v
   ```

6. From `Task_04`, run:

   ```bash
   python -m pytest -v
   ```

The pack contains complete replacement files for each changed module, its corresponding regression tests, and polished root/Task 02/Task 04 documentation. Files that did not need changes remain in your existing Git checkout.

## Files included

- `README.md`
- `Task_02/README.md`
- `Task_02/app/config.py`
- `Task_02/app/loaders/document_loader.py`
- `Task_02/app/ui.py`
- `Task_02/tests/test_document_loader.py`
- `Task_04/README.md`
- `Task_04/storage/sqlite_store.py`
- `Task_04/streamlit_app.py`
- `Task_04/frontend/sidebar.py`
- `Task_04/tests/test_storage.py`
