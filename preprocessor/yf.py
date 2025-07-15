import yfinance as yf
dat = yf.Ticker("MSFT")

dat = yf.Ticker("MSFT")
dat.info
dat.calendar
dat.analyst_price_targets
dat.quarterly_income_stmt
dat.history(period='1mo')
dat.option_chain(dat.options[0]).calls

df = yf.download(
    'MSFT',
    start="2025-01-01",
    end="2025-02-01",
    proxy=None,
    auto_adjust=False
)

print(df)