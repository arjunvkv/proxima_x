"""TradingView-style terminal trend showing tick WLS vs M5 bar WLS side-by-side."""
from io import StringIO
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress_bar import ProgressBar
from rich import box

from dashboard.nme.formatter import G, R, A, C, W, D


class BarStateDashboard:
    def __init__(self, width: int = 84):
        self.width = width

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _dir_arrow(val: int) -> str:
        return "▲" if val > 0 else ("▼" if val < 0 else "○")

    @staticmethod
    def _dir_clr(val: int):
        return G if val > 0 else (R if val < 0 else D)

    def render(self, bar_state: dict, bar_summary: str,
               currency_strengths: dict, strength_persistence: dict,
               cycle_count: int, forming_returns: Optional[dict] = None,
               pair_agreements: Optional[list] = None,
               terminal_trends: Optional[list] = None,
               max_mag: float = 1e-12) -> str:
        buf = StringIO()
        console = Console(file=buf, width=self.width, force_terminal=True,
                          color_system="truecolor", legacy_windows=False,
                          highlight=False)
        ready = bool(bar_state)

        # ── HEADER ─────────────────────────────────────────────
        status_clr = G if ready else A
        status_txt = "READY" if ready else "BUILD"
        header_t = Table.grid(padding=0)
        header_t.add_column()
        header_t.add_row(
            Text(" BAR STATE  ", style=f"bold {W}"),
            Text(f"CYCLE {cycle_count} ", style=C),
            Text("●", style=f"bold {status_clr}"),
            Text(f" {status_txt}  ", style=status_clr),
            Text(f"{bar_summary}", style=D),
        )
        console.print(Panel(header_t, box=box.SQUARE, border_style=G, padding=(0, 1)))

        # ── CURRENCY COMPARISON: TICK WLS vs BAR WLS ──────────
        comp_t = Table.grid(padding=(0, 1))
        comp_t.add_column()
        comp_t.add_column()
        comp_t.add_column()
        comp_t.add_column()

        ccy_order = (
            sorted(currency_strengths.keys(),
                   key=lambda c: abs(currency_strengths.get(c, 0)), reverse=True)
            if currency_strengths else []
        )

        comp_t.add_row(Text(" CCY ", style=f"bold {W}"),
                       Text(" TICK WLS (5s) ", style=f"bold {C}"),
                       Text("", style=D),
                       Text(" BAR WLS (M5) ", style=f"bold {A}"))

        for ccy in ccy_order:
            tick_val = currency_strengths.get(ccy, 0)
            sp_map = strength_persistence or {}
            tick_sp = sp_map.get(ccy, {})
            tick_dir = tick_sp.get("direction", 0)
            tick_arrow = self._dir_arrow(tick_dir)
            tick_clr = self._dir_clr(tick_dir)
            tick_bar_w = min(int(abs(tick_val) * 600), 12)
            tick_bar = ProgressBar(completed=tick_bar_w, total=12, width=6,
                                   complete_style=tick_clr, style=D)

            bar_s = bar_state.get(ccy, {})
            bar_dir = bar_s.get("direction", 0)
            bar_arrow = self._dir_arrow(bar_dir)
            bar_clr = self._dir_clr(bar_dir)
            bar_cons = bar_s.get("consistency", 0)
            bar_mom = bar_s.get("momentum", 0)
            bar_str = f" {bar_arrow}c={bar_cons:.2f}m={bar_mom:+.2f}"

            comp_t.add_row(
                Text(f" {ccy}", style=f"bold {W}"),
                Text(f" {tick_arrow} {tick_val:+.5f}", style=f"bold {tick_clr}"),
                tick_bar,
                Text(bar_str, style=f"{bar_clr}"),
            )

        console.print(Panel(comp_t, box=box.SQUARE, title=" Trend Comparison ", border_style=C, padding=(0, 1)))

        # ── PAIR AGREEMENT TABLE ──────────────────────────────
        if ready and pair_agreements:
            pair_t = Table(box=box.MINIMAL, header_style=f"bold {D}",
                           show_header=True, padding=(0, 1))
            pair_t.add_column("PAIR", style=f"bold {W}")
            pair_t.add_column("TICK")
            pair_t.add_column("BAR")
            pair_t.add_column("STATUS")

            for sym, tick_dir, bar_dir in pair_agreements[:6]:
                tick_arrow = self._dir_arrow(tick_dir)
                tick_clr = self._dir_clr(tick_dir)
                bar_arrow = self._dir_arrow(bar_dir)
                bar_clr = self._dir_clr(bar_dir)
                match = tick_dir == bar_dir and tick_dir != 0
                status_text = " MATCH" if match else " MISMATCH"
                status_clr = G if match else A if tick_dir != 0 else D

                pair_t.add_row(
                    sym,
                    Text(tick_arrow, style=f"bold {tick_clr}"),
                    Text(bar_arrow, style=f"bold {bar_clr}"),
                    Text(status_text, style=f"bold {status_clr}"),
                )

            if not pair_agreements:
                pair_t.add_row("--", Text("--", style=D),
                               Text("--", style=D), Text("--", style=D))

            console.print(Panel(pair_t, box=box.SQUARE,
                                title=" Tick vs Bar Agreement ",
                                border_style=A, padding=(0, 1)))

        # ── TERMINAL TREND (TradingView-style bars) ───────────
        if terminal_trends:
            tt = Table(box=box.MINIMAL, header_style=f"bold {D}",
                       show_header=True, padding=(0, 1))
            tt.add_column("PAIR", style=f"bold {W}")
            tt.add_column("TICK 5s", style=f"bold {C}")
            tt.add_column("BAR M5", style=f"bold {A}")
            tt.add_column("STATUS")

            m = max_mag if max_mag > 1e-12 else 1e-12
            bar_w = 10

            for sym, tick_seq, bar_seq, overall_dir, tick_val, bar_val in terminal_trends[:5]:
                def bar_text(val, clr):
                    pct = min(abs(val) / m, 1.0)
                    fill = int(round(pct * bar_w))
                    empty = bar_w - fill
                    block = "█" * fill + "░" * empty
                    return Text(f"{block} {val:+.5f}", style=f"bold {clr}")
                tick_b = bar_text(tick_val, G if tick_val > 0 else R)
                bar_b = bar_text(bar_val, G if bar_val > 0 else R)

                last_tick = tick_seq[-1] if tick_seq else 0
                last_bar = bar_seq[-1] if bar_seq else 0
                aligned = (last_tick > 0 and last_bar > 0) or (last_tick < 0 and last_bar < 0)
                status = Text(" ALIGNED", style=f"bold {G}") if aligned else (
                    Text(" CONFLICT", style=f"bold {A}") if last_tick != 0 else Text(" --", style=D))

                tt.add_row(sym, tick_b, bar_b, status)

            console.print(Panel(tt, box=box.SQUARE,
                                title=" TradingView Bars ",
                                border_style=G, padding=(0, 1)))

        # ── FORMING M5 RETURN (real-time) ─────────────────────
        if forming_returns:
            ft = Table.grid(padding=(0, 2))
            ft.add_column()
            ft.add_column()
            cells = []
            for sym, val in list(forming_returns.items())[:8]:
                clr = G if val > 1e-4 else (R if val < -1e-4 else D)
                cells.append(Text(f"{sym}", style=D))
                cells.append(Text(f"{val:+.5f}", style=f"bold {clr}"))
            if cells:
                ft.add_row(*cells)
            console.print(Panel(ft, box=box.SQUARE,
                                title=" Forming M5 Return (real-time) ",
                                border_style=C, padding=(0, 1)))

        return buf.getvalue()
