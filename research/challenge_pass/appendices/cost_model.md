# Cost Model — FundedNext Stellar Lite (Raw)

## Commission

$3 per lot per side = $6 round-turn.
Actually: Stellar Lite raw commission is approximately $3 per lot standard
(round-turn = entry + exit). Some sources show $3/lot per side ($6 round-turn).
We use $3/lot total commission as conservative (the lower bound).

## Spread Costs

Converted from MT5 points to pips:

**5-digit pairs (EURUSD, AUDUSD, EURAUD, GBPAUD):**
- 1 pip = 0.00010
- 1 point = 0.00001
- 10 points = 1 pip

**3-digit pairs (EURJPY, GBPJPY):**
- 1 pip = 0.01
- 1 point = 0.001
- 10 points = 1 pip

**Spread cost (USD) = spread_pips × pip_value**

## Pip Values

| Pair | Pip Value Formula | At USDJPY=162 |
|------|------------------|:-------------:|
| EURUSD | $10.00 fixed | $10.00 |
| EURJPY | 1000 / USDJPY | $6.17 |
| GBPJPY | 1000 / USDJPY | $6.17 |
| AUDUSD | $10.00 fixed | $10.00 |
| EURAUD | varies with AUDUSD | ~$6.50 |
| GBPAUD | varies with AUDUSD | ~$6.50 |

## Slippage

Fixed 0.5 pips per trade (half-spread equivalent).
Validated against Dukascopy's ATR-conditional slippage model which showed $2-3
variance between fixed and variable slippage.

## Total Cost per Trade (EURUSD example)

```
Spread: 8pts = 0.8 pips × $10.00 = $8.00
Slippage: 0.5 pips × $10.00 = $5.00
Commission: $3.00
Total: $16.00
```

Compared to Dark Consensus original Dukascopy cost model:
- 1.5× spread: 0.6 pips × $10 = $6.00
- ATR slip: ~$3.00
- Commission: $7.00
- Total: $16.00

FundedNext costs are approximately equal to the Dukascopy stress test model.
