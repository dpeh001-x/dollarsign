"""Reddit ticker scanner — finds trending stocks across finance subs.

Pulls recent /hot posts from r/wallstreetbets, r/stocks, r/investing, etc.
Extracts ticker mentions (matches $TICKER and bare TICKER), filters against
a known-tickers whitelist + an aggressive stop-word list to suppress false
positives like CEO/USA/FED. Scores each ticker by mention count and average
VADER sentiment.

Composite score = sqrt(mentions) * (1 + 0.3 * avg_sentiment) — favors
tickers with both high attention AND bullish lean, but doesn't completely
discount bearish-but-loud tickers (those can be shorting opportunities).

Auth: OAuth client_credentials. Needs env vars REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET (or Streamlit Cloud secrets). Without them, scan()
raises a clear RuntimeError pointing to setup docs.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass

import httpx
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

REDDIT_OAUTH_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"

DEFAULT_SUBS: tuple[str, ...] = (
    "wallstreetbets", "stocks", "investing", "options",
    "ValueInvesting", "SecurityAnalysis", "StockMarket",
    "DividendInvesting", "thetagang", "smallstreetbets",
)

# Stop-words: 1-5 char uppercase tokens that look like tickers but aren't.
# Aggressive on purpose — false positives are worse than misses for ranking.
TICKER_STOP_WORDS = frozenset({
    # Common acronyms
    "CEO","CFO","CTO","COO","CIO","CMO","IPO","SPO","USA","UK","EU","UN","NHS",
    "FED","FOMC","SEC","FBI","CIA","DOJ","FTC","FCC","FDA","DEA","IRS","DOD","DOE","DOL","DHS","DOT","HHS","VA",
    "GDP","CPI","PPI","PMI","PCE","NFP","JOLTS","ISM",
    "ATH","ATL","WSB","DD","TLDR","YOLO","FOMO","MOON","HODL","FUD","BTW",
    "IMO","IMHO","IIRC","AFAIK","TBH","TBF","ITT","RIP","LOL","LMAO","ROFL","WTF","OMG","BRB","FYI",
    "TY","NP","OK","OP","PM","AM","ET","PT","CT","MT","UTC","GMT","EST","PST","CST","MST",
    # Finance jargon (also tickers in some cases — explicitly excluded since they confuse the scanner)
    "ATM","OTM","ITM","DTE","IV","HV","RSI","MACD","SMA","EMA","ADX","ATR","BB","VWAP","PE","EPS",
    "ROI","ROE","ROIC","PEG","PB","PS","FCF","CAPEX","EBIT","EBITDA","SGA","COGS","DCF","NPV",
    "ARPU","CAC","LTV","MRR","ARR","SAAS","SOFR","LIBOR","NYSE","NASDAQ","DOW","SP500","SPX",
    # Currencies
    "USD","EUR","GBP","JPY","CNY","CAD","AUD","NZD","CHF","HKD","SGD","INR","MXN","KRW","SEK","NOK","DKK","RUB","BRL","ZAR",
    # English words 2-5 chars (subset — most common)
    "A","I","AND","ARE","BUT","CAN","DID","FOR","HAD","HAS","HOW","NOT","ONE","OUR","OUT","SAW","SEE","SHE","THE","TWO","WAS","WAY","WHO","WHY","YOU",
    "ALL","ANY","BAD","BIG","BOY","BUY","CAR","DAY","EAT","END","EYE","FAR","FEW","FUN","GET","GOT","GUY","HER","HIM","HIS","HOT","ITS","LET","LOT","LOW","MAN","MAP","MAY","NEW","NOW","ODD","OFF","OLD","OWN","PAY","PER","PUT","RUN","SAT","SAY","SET","SIT","TOP","TRY","USE","WIN","YES","YET",
    "ABLE","AREA","AWAY","BACK","BANK","BASE","BEAT","BEEN","BEER","BEST","BILL","BOOK","BOTH","BOSS","BUSY","CALL","CAME","CARD","CARE","CASE","CASH","CAST","CHAT","CITY","CLUB","COIN","COLD","COME","COOK","COOL","COPE","COPY","CORE","COST","DARK","DATA","DATE","DEAD","DEAL","DEAR","DEBT","DEEP","DESK","DOES","DONE","DOOR","DOWN","DRAW","DROP","EACH","EARN","EASE","EAST","EASY","EDGE","ELSE","EVEN","EVER","EVIL","EXIT","FACE","FACT","FAIL","FAIR","FALL","FAST","FATE","FEAR","FEED","FEEL","FEET","FELL","FELT","FILE","FILL","FILM","FIND","FINE","FIRE","FIRM","FISH","FIVE","FLAT","FLOW","FOOD","FOOT","FORM","FOUR","FREE","FROM","FUEL","FULL","FUND","GAIN","GAME","GAVE","GIRL","GIVE","GOAL","GOES","GOLD","GOLF","GONE","GOOD","GREW","GROW","HAIR","HALF","HALL","HAND","HANG","HARD","HARM","HATE","HAVE","HEAD","HEAR","HEAT","HELD","HELL","HELP","HERE","HIGH","HILL","HIRE","HOLD","HOLE","HOLY","HOME","HOPE","HORN","HOST","HOUR","HUGE","HURT","IDEA","INCH","INTO","IRON","ITEM","JOIN","JUMP","JURY","JUST","KEEP","KEPT","KICK","KIDS","KILL","KIND","KING","KNEE","KNEW","KNOW","LACK","LADY","LAKE","LAND","LANE","LAST","LATE","LEAD","LEAN","LEFT","LESS","LIFE","LIFT","LIKE","LINE","LINK","LION","LIST","LIVE","LOAD","LOAN","LOCK","LONG","LOOK","LORD","LOSE","LOSS","LOST","LOTS","LOUD","LOVE","LUCK","MADE","MAIL","MAIN","MAKE","MALE","MANY","MARK","MASS","MEAL","MEAN","MEAT","MEET","MENU","MERE","MILD","MILE","MILK","MIND","MINE","MISS","MOLD","MOON","MORE","MOST","MOVE","MUCH","MUST","NAIL","NAME","NEAR","NECK","NEED","NEWS","NEXT","NICE","NINE","NODE","NONE","NOON","NORM","NOSE","NOTE","ONCE","ONLY","ONTO","OPEN","ORAL","OURS","OVAL","OVEN","OVER","OWNS","PACE","PACK","PAGE","PAID","PAIN","PAIR","PALE","PARK","PART","PASS","PAST","PATH","PEAK","PEER","PICK","PILE","PINK","PIPE","PLAN","PLAY","PLOT","PLUS","POEM","POLE","POLL","POND","POOL","POOR","PORT","POST","POUR","PULL","PURE","PUSH","QUIT","RACE","RAGE","RAID","RAIL","RAIN","RANK","RARE","RATE","READ","REAL","RELY","RENT","REST","RICE","RICH","RIDE","RING","RISE","RISK","ROAD","ROCK","ROLE","ROLL","ROOF","ROOM","ROOT","ROPE","ROSE","RULE","RUSH","SACK","SAFE","SAID","SAIL","SAKE","SALE","SALT","SAME","SAND","SAVE","SCAN","SEAT","SEED","SEEK","SEEM","SEEN","SELF","SELL","SEND","SENT","SHED","SHIP","SHOE","SHOP","SHOT","SHOW","SHUT","SICK","SIDE","SIGN","SILK","SING","SINK","SITE","SIZE","SKIN","SLAM","SLAP","SLOW","SNAP","SNOW","SOAP","SOFT","SOIL","SOLD","SOLE","SOME","SONG","SOON","SORT","SOUL","SOUP","SPED","SPIN","SPOT","STAR","STAY","STEM","STEP","STOP","SUCH","SUIT","SUNG","SURE","TAKE","TALE","TALK","TALL","TAME","TANK","TAPE","TASK","TAXI","TEAM","TEAR","TECH","TELL","TEND","TENT","TERM","TEST","TEXT","THAN","THAT","THEM","THEN","THEY","THIN","THIS","THUS","TIDE","TIDY","TIED","TIES","TILE","TIME","TINY","TIRE","TOLD","TONE","TOOK","TOOL","TOPS","TORE","TORN","TOSS","TOUR","TOWN","TREE","TRIM","TRIP","TRUE","TUBE","TUNE","TURN","TYPE","UGLY","UNIT","UPON","USED","USER","USES","VARY","VAST","VERY","VICE","VIEW","VOID","VOTE","WAIT","WAKE","WALK","WALL","WANT","WARM","WARN","WARS","WASH","WAVE","WAYS","WEAK","WEAR","WEEK","WELL","WENT","WERE","WEST","WHAT","WHEN","WIDE","WIFE","WILD","WILL","WIND","WINE","WING","WIPE","WIRE","WISE","WISH","WITH","WOOD","WOOL","WORD","WORE","WORK","WORM","WORN","YARD","YEAR","ZERO","ZONE",
})

# Whitelist of valid tickers — covers S&P 500 + popular ETFs + meme stocks + crypto.
# False positives suppressed by intersection with this set; new tickers added here.
KNOWN_TICKERS = frozenset({
    # Mega-cap & MAG7
    "AAPL","MSFT","GOOGL","GOOG","AMZN","META","NVDA","TSLA","AVGO","BRKB","BRK.B","ORCL",
    # Large-cap
    "LLY","JPM","V","WMT","MA","XOM","JNJ","PG","HD","COST","CVX","ABBV","KO","PEP","BAC","ADBE","CRM","MRK","NFLX","TMO","ACN","CSCO","ABT","DIS","WFC","MCD","DHR","INTC","AMD","INTU","NKE","TXN","PFE","CMCSA","UNH","UNP","HON","UPS","T","QCOM","GS","BA","RTX","LIN","SBUX","BLK","SCHW","NEE","C","AMGN","MS","COP","BX","MMM","CAT","DE","SPGI","BMY","NOW","BKNG","GE","ADP","CB","MO","PLD","ETN","ELV","SYK","CI","MDT","TGT","ZTS","LMT","ISRG","BSX","REGN","VRTX","DUK","PNC","SO","MDLZ","SHW","FI","CL","TJX","BDX","EW","AMT","CME","ITW","NSC","FDX","SLB","HUM","APD","MU","KLAC","SNPS","LRCX","FCX","MAR","MCK","CMG","WM","AON","ICE","PSX","EOG","ECL","ADI","TFC","CSX","ETR","COF","TRV","BIIB","HCA","NOC","GD","GIS","KMB","DG","DLR","PSA","MMC","HLT","D","AIG","EQIX","SRE","WELL","KDP","FIS","MCHP","WBD","COR","PGR","AEP","NXPI","STZ","AZO","ORLY","ROP","PRU","CTSH","KHC","FTNT","MET","ROST","KR","JCI","TT","RSG","PCAR","DLTR","PCG","XEL","WMB","WBA","HSY","BK","HES","AFL","ED","ALL","HPQ","MNST","DXCM","TRGP","HAL","DOW","KMI","STT","DD","FDS","DAL","UAL","LUV","AAL",
    # Tech / fintech / growth
    "PYPL","SQ","ROKU","SHOP","SPOT","ZM","DOCU","UBER","LYFT","ABNB","DASH","COIN","HOOD","RBLX","SNOW","DDOG","NET","CRWD","ZS","OKTA","TEAM","TWLO","PINS","SNAP","X","MSTR","PLTR",
    # Meme stocks
    "GME","AMC","BB","NOK","BBBY","WISH","CLOV","SOFI","RIOT","MARA","CIFR","BTBT",
    # Chinese
    "BABA","BIDU","JD","PDD","TCEHY","NIO","XPEV","LI",
    # International large
    "ASML","TSM","SAP","SHEL","BP","TM","SNY","NVS","UL","HSBC","BTI","RIO","BHP",
    # Major ETFs
    "SPY","QQQ","IWM","DIA","VOO","VTI","IVV","VEA","VWO","EFA","EEM","BND","AGG","HYG","LQD","TIP","GLD","SLV","USO","UNG","DBC","DBA","TLT","UVXY","VXX","VIXY","SVXY",
    # Leveraged/inverse
    "TQQQ","SQQQ","SOXL","SOXS","TNA","TZA","FAS","FAZ","SPXL","SPXS","SPXU","UPRO",
    # Sector ETFs
    "XLF","XLE","XLK","XLV","XLY","XLP","XLI","XLU","XLB","XLRE","XLC","XBI","SMH","SOXX","KWEB","KRE","ITB","XHB","ITA","XME","GDX","GDXJ","SLX","REMX","COPX","URA","TAN","ICLN","PBW","FAN","HACK","BOTZ","ROBO","JETS",
    # ARKK family
    "ARKK","ARKQ","ARKG","ARKW","ARKF",
    # Crypto (yfinance-style symbols)
    "BTC-USD","ETH-USD","SOL-USD","DOGE-USD","ADA-USD","AVAX-USD","XRP-USD","DOT-USD","SHIB-USD","MATIC-USD",
    # Crypto bare (handled specially — check both forms)
    "BTC","ETH","SOL","DOGE","ADA","AVAX","XRP",
})

# Match $TICKER (preferred) or bare 1-5 char uppercase token, optional .SUFFIX
TICKER_PATTERN = re.compile(r"\$?([A-Z]{1,5}(?:\.[A-Z])?)\b")

_analyzer = SentimentIntensityAnalyzer()
_token_cache: dict[str, float | str | None] = {"token": None, "expires_at": 0.0}


def _is_configured() -> bool:
    return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))


def _get_token(client: httpx.Client) -> str | None:
    now = time.time()
    if _token_cache["token"] and float(_token_cache["expires_at"]) > now + 60:
        return str(_token_cache["token"])

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    user_agent = os.getenv("REDDIT_USER_AGENT", "dollarsign-scanner/0.1")
    resp = client.post(
        REDDIT_OAUTH_URL,
        auth=(client_id, client_secret),
        data={"grant_type": "client_credentials"},
        headers={"User-Agent": user_agent},
    )
    resp.raise_for_status()
    body = resp.json()
    _token_cache["token"] = body.get("access_token")
    _token_cache["expires_at"] = now + body.get("expires_in", 3600)
    return str(_token_cache["token"])


def _extract_tickers(text: str) -> set[str]:
    if not text:
        return set()
    candidates = set(TICKER_PATTERN.findall(text.upper()))
    valid = set()
    for t in candidates:
        if t in TICKER_STOP_WORDS:
            continue
        # Map bare crypto symbols to yfinance form so the ML pipeline can use them
        if t in ("BTC", "ETH", "SOL", "DOGE", "ADA", "AVAX", "XRP"):
            valid.add(f"{t}-USD")
        elif t in KNOWN_TICKERS:
            valid.add(t)
    return valid


@dataclass
class TickerSignal:
    ticker: str
    mentions: int
    sentiment_avg: float       # -1..1
    sentiment_n: int
    score: float               # composite ranking
    sample_post: str
    sample_url: str
    subs: tuple[str, ...]      # which subs mentioned it


def scan(
    subreddits: tuple[str, ...] = DEFAULT_SUBS,
    posts_per_sub: int = 50,
    max_age_hours: float = 48.0,
) -> list[TickerSignal]:
    """Scan recent /hot posts. Returns ranked TickerSignal list (highest score first)."""
    user_agent = os.getenv("REDDIT_USER_AGENT", "dollarsign-scanner/0.1")

    mentions: dict[str, int] = defaultdict(int)
    sentiments: dict[str, list[float]] = defaultdict(list)
    sample_posts: dict[str, tuple[int, str, str]] = {}  # ticker -> (post_score, title, url)
    subs_per_ticker: dict[str, set[str]] = defaultdict(set)

    cutoff = time.time() - max_age_hours * 3600

    with httpx.Client(timeout=30.0, headers={"User-Agent": user_agent}) as client:
        token = _get_token(client)
        if not token:
            raise RuntimeError(
                "Reddit credentials not set. Add REDDIT_CLIENT_ID and "
                "REDDIT_CLIENT_SECRET to your .env (locally) or Streamlit Cloud "
                "secrets (deployed). Create the app at https://www.reddit.com/prefs/apps "
                "(type 'script', any redirect)."
            )
        client.headers["Authorization"] = f"Bearer {token}"

        for sub in subreddits:
            try:
                resp = client.get(
                    f"{REDDIT_API}/r/{sub}/hot",
                    params={"limit": posts_per_sub},
                )
                if resp.status_code != 200:
                    logger.warning("scan: r/%s returned %d", sub, resp.status_code)
                    continue
                data = resp.json()
            except Exception as e:
                logger.warning("scan: r/%s failed: %s", sub, e)
                continue

            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                if post.get("created_utc", 0) < cutoff:
                    continue

                title = post.get("title", "") or ""
                body = post.get("selftext", "") or ""
                full_text = f"{title}\n{body}"
                tickers = _extract_tickers(full_text)
                if not tickers:
                    continue

                sent = _analyzer.polarity_scores(full_text[:1024])
                compound = sent["compound"]
                post_score = int(post.get("score") or 0)
                post_url = f"https://reddit.com{post.get('permalink', '')}"

                for t in tickers:
                    mentions[t] += 1
                    sentiments[t].append(compound)
                    subs_per_ticker[t].add(sub)
                    if t not in sample_posts or post_score > sample_posts[t][0]:
                        sample_posts[t] = (post_score, title, post_url)

    results: list[TickerSignal] = []
    for ticker, n in mentions.items():
        if not sentiments[ticker]:
            continue
        avg_sent = sum(sentiments[ticker]) / len(sentiments[ticker])
        score = (n ** 0.5) * (1.0 + 0.3 * avg_sent)
        sample = sample_posts.get(ticker, (0, "", ""))
        results.append(TickerSignal(
            ticker=ticker,
            mentions=n,
            sentiment_avg=round(avg_sent, 3),
            sentiment_n=len(sentiments[ticker]),
            score=round(score, 3),
            sample_post=sample[1][:140],
            sample_url=sample[2],
            subs=tuple(sorted(subs_per_ticker[ticker])),
        ))

    return sorted(results, key=lambda r: r.score, reverse=True)


if __name__ == "__main__":
    import sys
    try:
        results = scan(posts_per_sub=30, max_age_hours=72)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(results)} unique tickers in recent finance Reddit:\n")
    print(f"{'rank':>4} {'ticker':<10} {'mentions':>9} {'sent':>7} {'score':>7}  sample post")
    print(f"{'-'*4} {'-'*10} {'-'*9} {'-'*7} {'-'*7}  {'-'*60}")
    for i, r in enumerate(results[:25], 1):
        sent_str = f"{r.sentiment_avg:+.2f}"
        print(f"{i:>4} {r.ticker:<10} {r.mentions:>9} {sent_str:>7} {r.score:>7.2f}  {r.sample_post[:60]}")
