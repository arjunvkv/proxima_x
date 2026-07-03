MIN_TRADES_BOOTSTRAP = 3
MIN_TRADES_COLLECTING = 10
MIN_TRADES_EARLY = 25
MIN_TRADES_INTERMEDIATE = 50
MIN_TRADES_FULL = 50

MIN_DAYS_FOR_ANNUALIZATION = 2


def evidence_phase(n_trades: int) -> str:
    if n_trades < MIN_TRADES_BOOTSTRAP:
        return "BOOTSTRAP"
    elif n_trades < MIN_TRADES_COLLECTING:
        return "COLLECTING_EVIDENCE"
    elif n_trades < MIN_TRADES_EARLY:
        return "EARLY_VALIDATION"
    elif n_trades < MIN_TRADES_INTERMEDIATE:
        return "INTERMEDIATE_VALIDATION"
    else:
        return "FULL_VALIDATION"


def has_min_sample(n_trades: int) -> bool:
    return n_trades >= MIN_TRADES_COLLECTING


def annualization_valid(days_elapsed: int) -> bool:
    return days_elapsed >= MIN_DAYS_FOR_ANNUALIZATION
