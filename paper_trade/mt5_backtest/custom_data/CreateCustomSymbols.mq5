#property script_show_inputs

input string DataDir = "custom_data"; // Subfolder under Files/

string PAIR_NAMES[3] = {"FN_EURJPY", "FN_EURUSD", "FN_GBPJPY"};
string CSV_NAMES[3]  = {"FN_EURJPY.csv", "FN_EURUSD.csv", "FN_GBPJPY.csv"};
int    SYM_DIGITS[3] = {3, 5, 3};

void OnStart() {
   int total_imported = 0;
   for(int i = 0; i < 3; i++) {
      string sym = PAIR_NAMES[i];
      
      // Create custom symbol (ignore error if already exists)
      long create_result = CustomSymbolCreate(sym, "Custom\\FundedNext");
      if(create_result == -1) {
         int e = GetLastError();
         if(e == 6604 || e == 6603) {
            Print(sym + " already exists, updating...");
         } else {
            Print("CustomSymbolCreate(" + sym + ") failed error=" + IntegerToString(e) + " (may already exist)");
         }
      } else {
         Print("Created custom symbol: " + sym);
      }
      
      // Set symbol properties
      if(!CustomSymbolSetInteger(sym, SYMBOL_DIGITS, SYM_DIGITS[i]))
         Print("  SYMBOL_DIGITS set failed: " + IntegerToString(GetLastError()));
      if(!CustomSymbolSetInteger(sym, SYMBOL_TRADE_CALC_MODE, 0))
         Print("  SYMBOL_TRADE_CALC_MODE set failed: " + IntegerToString(GetLastError()));
      if(!CustomSymbolSetDouble(sym, SYMBOL_TRADE_CONTRACT_SIZE, 100000))
         Print("  SYMBOL_TRADE_CONTRACT_SIZE set failed: " + IntegerToString(GetLastError()));
      if(!CustomSymbolSetInteger(sym, SYMBOL_TRADE_MODE, SYMBOL_TRADE_MODE_DISABLED))
         Print("  SYMBOL_TRADE_MODE set failed: " + IntegerToString(GetLastError()));
      
      // Read CSV and update bars
      string path = DataDir + "\\" + CSV_NAMES[i];
      int fh = FileOpen(path, FILE_READ | FILE_CSV | FILE_ANSI, ",");
      if(fh == INVALID_HANDLE) {
         Print("  Cannot open " + path + " (error " + IntegerToString(GetLastError()) + ")");
         Print("  Place CSV files in " + TerminalInfoString(TERMINAL_DATA_PATH) + "\\MQL5\\Files\\" + DataDir + "\\");
         continue;
      }
      
      // Read all bars into array
      MqlRates rates[];
      int n = 0;
      while(!FileIsEnding(fh)) {
         string date_s = FileReadString(fh);
         string time_s = FileReadString(fh);
         string open_s  = FileReadString(fh);
         string high_s  = FileReadString(fh);
         string low_s   = FileReadString(fh);
         string close_s = FileReadString(fh);
         string vol_s   = FileReadString(fh);
         
         if(date_s == "" || time_s == "") continue;
         
         ArrayResize(rates, n + 1);
         rates[n].time = StringToTime(date_s + " " + time_s);
         rates[n].open  = StringToDouble(open_s);
         rates[n].high  = StringToDouble(high_s);
         rates[n].low   = StringToDouble(low_s);
         rates[n].close = StringToDouble(close_s);
         rates[n].tick_volume   = (long)StringToInteger(vol_s);
         rates[n].spread        = 0;
         rates[n].real_volume   = 0;
         n++;
      }
      FileClose(fh);
      
      if(n == 0) {
         Print("  No bars read from " + path);
         continue;
      }
      
      // Update in chunks of 10000 to avoid memory issues
      int chunk = 10000;
      int imported = 0;
      for(int start = 0; start < n; start += chunk) {
         int end = MathMin(start + chunk, n);
         MqlRates chunk_rates[];
         ArrayCopy(chunk_rates, rates, 0, start, end - start);
         int code = CustomRatesUpdate(sym, chunk_rates);
         if(code < 0) {
            Print("  CustomRatesUpdate error=" + IntegerToString(GetLastError()) + " at bar " + IntegerToString(start));
         } else {
            imported += code;
         }
      }
      total_imported += imported;
      Print("  Imported " + IntegerToString(imported) + " bars into " + sym);
      
      // Select symbol in Market Watch
      SymbolSelect(sym, true);
   }
   
   Print("Total: " + IntegerToString(total_imported) + " bars imported across 3 custom symbols");
   Print("DONE. Restart terminal, then verify: drag each symbol onto a chart.");
}
