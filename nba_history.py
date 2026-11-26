"""Comando e importações compatíveis do histórico de jogos."""
from src.data.history import (
    ARCHIVE,
    COLUMNS,
    DEFAULT_DB,
    fetch_archive_season,
    fetch_season,
    load_logs,
    main,
    normalize_logs,
    read_archive_parquet,
    save_logs,
)


if __name__ == "__main__":
    main()
