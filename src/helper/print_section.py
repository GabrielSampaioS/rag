_last_line_blank = False  # evita linhas em branco duplicadas


def _line(text=""):
    global _last_line_blank
    print(text)
    _last_line_blank = False


def print_blank():
    """Pula uma linha, só se a última linha impressa não estiver em branco."""
    global _last_line_blank
    if not _last_line_blank:
        print()
        _last_line_blank = True


def print_section(title):
    print_blank()
    _line("=" * 50)
    _line(title.center(50))
    _line("=" * 50)
    print_blank()


def print_subsection(title):
    print_blank()
    _line(title)
    print_blank()


def print_info(message):
    _line(f"[INFO] {message}")


def print_success(message):
    _line(f"[SUCESSO] {message}")


def print_warning(message):
    _line(f"[AVISO] {message}")


def print_error(message):
    _line(f"[ERRO] {message}")