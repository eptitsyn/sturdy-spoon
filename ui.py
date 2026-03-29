"""Console helpers with Rich support and plain-text fallback."""

from __future__ import annotations

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:  # pragma: no cover - fallback when rich is unavailable
    box = None
    Console = None
    Panel = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    MofNCompleteColumn = None
    TimeElapsedColumn = None
    Table = None
    RICH_AVAILABLE = False
    console = None


def print_section(title: str) -> None:
    if RICH_AVAILABLE:
        console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="cyan"))
        return
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_info(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(message)
        return
    print(message)


def print_success(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[green]{message}[/green]")
        return
    print(message)


def print_warning(message: str) -> None:
    if RICH_AVAILABLE:
        console.print(f"[yellow]{message}[/yellow]")
        return
    print(message)

