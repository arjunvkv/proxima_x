#property version "1.00"

int OnInit() {
   Print("TERMINAL_PATH: " + TerminalInfoString(TERMINAL_PATH));
   Print("TERMINAL_COMMONDATA_PATH: " + TerminalInfoString(TERMINAL_COMMONDATA_PATH));
   Print("TERMINAL_DATA_PATH: " + TerminalInfoString(TERMINAL_DATA_PATH));

   int h;
   h = FileOpen("impulse_ticks.bin", FILE_BIN | FILE_READ);
   Print("Open(impulse_ticks.bin): " + IntegerToString(h));
   if (h != INVALID_HANDLE) FileClose(h);

   h = FileOpen("Files\\impulse_ticks.bin", FILE_BIN | FILE_READ);
   Print("Open(Files/impulse_ticks.bin): " + IntegerToString(h));
   if (h != INVALID_HANDLE) FileClose(h);

   h = FileOpen("..\\MQL5\\Files\\impulse_ticks.bin", FILE_BIN | FILE_READ);
   Print("Open(../MQL5/Files/impulse_ticks.bin): " + IntegerToString(h));
   if (h != INVALID_HANDLE) FileClose(h);

   h = FileOpen("..\\..\\MQL5\\Files\\impulse_ticks.bin", FILE_BIN | FILE_READ);
   Print("Open(../../MQL5/Files/impulse_ticks.bin): " + IntegerToString(h));
   if (h != INVALID_HANDLE) FileClose(h);

   h = FileOpen("test_write.txt", FILE_TXT | FILE_WRITE);
   if (h != INVALID_HANDLE) {
      Print("Can write, path: " + TerminalInfoString(TERMINAL_PATH));
      FileWrite(h, "hello");
      FileClose(h);
   } else {
      Print("Cannot write");
   }

   return INIT_FAILED;
}

void OnTick() { ExpertRemove(); }
