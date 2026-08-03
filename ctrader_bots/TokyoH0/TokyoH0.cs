using System;
using System.Collections.Generic;
using System.Linq;
using cAlgo.API;
using cAlgo.API.Internals;

namespace cAlgo.Robots
{
    [Robot(TimeZone = TimeZones.UTC, AccessRights = AccessRights.FullAccess)]
    public class TokyoH0 : Robot
    {
        [Parameter("Lookback Bars", Group = "Strategy", DefaultValue = 6)]
        public int LookbackBars { get; set; }

        [Parameter("Hold Bars", Group = "Strategy", DefaultValue = 12)]
        public int HoldBars { get; set; }

        [Parameter("Top N Pairs", Group = "Strategy", DefaultValue = 5)]
        public int TopN { get; set; }

        [Parameter("Session Hour (UTC)", Group = "Strategy", DefaultValue = 0)]
        public int SessionHour { get; set; }

        [Parameter("Min Valid Pairs", Group = "Strategy", DefaultValue = 8)]
        public int MinPairs { get; set; }

        [Parameter("Base Lot", Group = "Risk", DefaultValue = 0.1)]
        public double BaseLot { get; set; }

        [Parameter("Min Confidence", Group = "Risk", DefaultValue = 0.3)]
        public double MinConfidence { get; set; }

        [Parameter("Min Volatility", Group = "Risk", DefaultValue = 0.001)]
        public double MinVol { get; set; }

        private static readonly string[] Pairs =
        {
            "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "EURJPY",
            "GBPJPY", "EURAUD", "EURNZD", "GBPAUD", "GBPNZD",
            "GBPCAD", "AUDNZD", "USDCAD", "NZDUSD", "EURGBP",
            "EURCHF", "USDCHF", "AUDJPY"
        };

        private const string Label = "TokyoH0";
        private readonly Dictionary<string, int> _entryBar = new Dictionary<string, int>();
        private readonly Dictionary<string, double[]> _volBuf = new Dictionary<string, double[]>();
        private readonly Dictionary<string, int> _volCount = new Dictionary<string, int>();
        private DateTime _lastEntryDate = DateTime.MinValue;
        private int _barCount;

        protected override void OnStart()
        {
            Bars.BarOpened += OnBarOpened;
            Print("TokyoH0 started: lb={0} hold={1} n={2}", LookbackBars, HoldBars, TopN);
        }

        private void OnBarOpened(BarOpenedEventArgs args)
        {
            _barCount++;
            CheckExits();
            CheckEntry();
        }

        private double GetVolatility(string symbol, double ret)
        {
            if (!_volCount.ContainsKey(symbol))
            {
                _volCount[symbol] = 0;
                _volBuf[symbol] = new double[10];
            }

            if (_volCount[symbol] < 10)
            {
                _volCount[symbol]++;
                return MinVol;
            }

            double absRet = Math.Abs(ret);
            var buf = _volBuf[symbol];
            for (int i = 0; i < 9; i++) buf[i] = buf[i + 1];
            buf[9] = absRet;
            return buf.Average();
        }

        private void CheckEntry()
        {
            var now = Bars.LastBar.OpenTime;
            if (now.Hour != SessionHour || now.Minute != 0)
                return;

            var today = now.Date;
            if (today == _lastEntryDate)
                return;
            _lastEntryDate = today;

            int lb = Math.Max(2, LookbackBars);
            var candidates = new List<(string Symbol, double Return)>();

            foreach (var symbol in Pairs)
            {
                var bars = MarketData.GetBars(TimeFrame.Minute5, symbol);
                if (bars == null || bars.ClosePrices.Count <= lb + 2)
                    continue;

                int idx = bars.ClosePrices.Count - 1;
                double cur = bars.ClosePrices[idx];
                double prv = bars.ClosePrices[idx - lb];
                if (cur <= 0 || prv <= 0)
                    continue;

                double ret = Math.Log(cur / prv);

                double pv = bars.ClosePrices[idx - 1];
                if (pv > 0)
                {
                    double gp = Math.Abs(cur - pv) / pv * 100.0;
                    if (gp >= 0.5) continue;
                }

                int cb = Math.Min(3, lb / 2);
                if (cb >= 2)
                {
                    double ps = bars.ClosePrices[idx - cb];
                    if (ps > 0 && Math.Log(cur / ps) > 0) continue;
                }

                candidates.Add((symbol, ret));
            }

            if (candidates.Count < MinPairs)
            {
                Print("SKIP {0} < {1}", candidates.Count, MinPairs);
                return;
            }

            candidates.Sort((a, b) => a.Return.CompareTo(b.Return));
            int take = Math.Min(TopN, candidates.Count);
            Print("ENTRY v={0} top={1}", candidates.Count, take);

            int entered = 0;
            for (int i = 0; i < take; i++)
            {
                if (candidates[i].Return >= 0) break;

                var (symbol, ret) = candidates[i];
                double vol = GetVolatility(symbol, ret);
                double margin = Math.Abs(ret) / Math.Max(vol, 1e-10);
                double conf = Math.Min(0.95, margin * 0.15);

                if (conf < MinConfidence)
                {
                    Print("  SKIP {0} conf={1:F4}", symbol, conf);
                    continue;
                }

                long volume = (long)Symbol.QuantityToVolumeInUnits(BaseLot);
                var result = ExecuteMarketOrder(TradeType.Buy, symbol, volume, Label, 500, null, "TokyoH0");
                if (result.IsSuccessful)
                {
                    _entryBar[symbol] = _barCount;
                    entered++;
                    Print("  OPEN {0} @{1}", symbol, result.Position.EntryPrice);
                }
            }

            Print("Entered {0}/{1}", entered, take);
        }

        private void CheckExits()
        {
            var toClose = new List<string>();
            foreach (var kvp in _entryBar)
            {
                if (_barCount - kvp.Value >= HoldBars)
                    toClose.Add(kvp.Key);
            }

            foreach (var symbol in toClose)
            {
                var pos = Positions.Find(Label, symbol);
                if (pos != null)
                {
                    ClosePosition(pos);
                    Print("  CLOSE {0} expiry pnl={1:F2}", symbol, pos.NetProfit);
                }
                _entryBar.Remove(symbol);
            }
        }

        protected override void OnStop()
        {
            Print("TokyoH0 stopped");
        }
    }
}
