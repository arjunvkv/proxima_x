//+------------------------------------------------------------------+
//| Check_Account_Permissions.mq5                                   |
//| Diagnostic script to check server trade permission flags          |
//+------------------------------------------------------------------+
#property copyright "Proxima Trading"
#property version   "1.00"
#property script_show_inputs

void OnStart() {
   Print("=================================================");
   Print("FUNDEDNEXT SERVER ACCOUNT DIAGNOSTIC REPORT");
   Print("=================================================");
   Print("Account Login Number : ", AccountInfoInteger(ACCOUNT_LOGIN));
   Print("Account Server Name  : ", AccountInfoString(ACCOUNT_SERVER));
   Print("Account Company Name : ", AccountInfoString(ACCOUNT_COMPANY));
   Print("Account Name         : ", AccountInfoString(ACCOUNT_NAME));
   Print("Account Currency     : ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Account Balance      : $", AccountInfoDouble(ACCOUNT_BALANCE));
   
   bool trade_allowed = (bool)AccountInfoInteger(ACCOUNT_TRADE_ALLOWED);
   bool trade_expert  = (bool)AccountInfoInteger(ACCOUNT_TRADE_EXPERT);
   int  trade_mode    = (int)AccountInfoInteger(ACCOUNT_TRADE_MODE);
   
   Print("ACCOUNT_TRADE_ALLOWED : ", trade_allowed ? "TRUE (Trade Allowed)" : "FALSE (BLOCKED BY SERVER)");
   Print("ACCOUNT_TRADE_EXPERT  : ", trade_expert ? "TRUE (EAs Allowed)" : "FALSE (EAS BLOCKED BY SERVER)");
   Print("ACCOUNT_TRADE_MODE    : ", trade_mode);
   Print("=================================================");
}
